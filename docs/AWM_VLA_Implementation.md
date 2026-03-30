# AWM-VLA (Advantage Weighted Matching for VLA) — 实现文档

## 1. 算法概述

AWM-VLA 的核心思想：**用归一化 advantage 直接作为 flow matching loss 的权重（可以为负！）**。

```
L_total = Ã · ||v_θ - u||²  +  β · ||v_θ - v_old||²  +  α · L_V
          ↑ AWM loss (双向信号)  ↑ trust region           ↑ value training
```

关键创新：**Ã 可以为负**，这是和 FPI（exp(A/λ) > 0 永远为正）的根本区别：

- Ã > 0（好的 action chunk）→ 梯度拉 v_θ **靠近**好 action 的 target → 学习好动作
- Ã < 0（差的 action chunk）→ 梯度推 v_θ **远离**差 action 的 target → 主动遗忘差动作
- Ã ≈ 0（平均水平）→ 贡献约为零 → 既不拉也不推

数学上等价于 policy gradient，但方差更低（在 clean action 上计算，而非 noisy intermediate）。

## 2. 数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Rollout Worker                                │
│  ┌──────────────┐                                                       │
│  │ π_θ rollout   │──→ (o_t, a_t, r_t, prev_values) per step            │
│  └──────────────┘                                                       │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ EMA model (_compute_awm_annotations, SINGLE VLM forward)    │       │
│  │   forward_velocity(return_value=True):                       │       │
│  │   ├→ v_old (trust region anchor)                             │       │
│  │   └→ V_target(o_t) (TD bootstrap, detached)                 │       │
│  │   + pre-compute (t, ε) for training                          │       │
│  └──────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  send to actor
┌──────────────────────────────────────────────────────────────────────────┐
│                           Actor Worker                                   │
│                                                                          │
│  ┌─── Credit Assignment ───────────────────────────────────────────┐    │
│  │  A_t = V(o_{t+1}) - V(o_t)             (raw advantage)         │    │
│  │  Ã_t = clip((A_t - μ) / (σ + ε), -c, c)  (normalize + clip)  │    │
│  │  td_target = r_t + γ · V_target(o_{t+1})                      │    │
│  │  progress_target = (t/T) · R            (reward-scaled, π-RL)  │    │
│  │  [warmup: Ã=0 for first N epochs]                              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│        │                                                                │
│        ▼                                                                │
│  ┌─── Training (per mini-batch, SINGLE VLM forward) ─────────────┐    │
│  │                                                                 │    │
│  │  ForwardType.VELOCITY(return_value=True):                       │    │
│  │    VLM prefix encoding → [一次 VLM forward]                     │    │
│  │    ├→ suffix → velocity head → v_θ   (有梯度, 训 policy)        │    │
│  │    └→ [detach] → value head → V(o_t)  (无梯度穿透 VLM)         │    │
│  │                                                                 │    │
│  │  Value Loss:                                                    │    │
│  │    L_TD = ||V_θ(o_t) - td_target||²                            │    │
│  │    L_prog = ||V_θ(o_t) - progress_target||²  (all trajectories)│    │
│  │    L_V = L_TD + L_prog                                          │    │
│  │                                                                 │    │
│  │  Policy Loss (pre-computed t, ε from rollout):                  │    │
│  │    u = ε - a  (flow matching target)                            │    │
│  │    L_AWM = Ã · ||v_θ - u||²    (Ã can be negative!)           │    │
│  │    L_KL  = ||v_θ - v_old||²   (trust region)                   │    │
│  │                                                                 │    │
│  │  Combined: L = L_AWM + α · L_V + β · L_KL                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. AWM vs FPI：为什么 FPI 失败而 AWM 可能成功

| 特性 | FPI (失败) | AWM-VLA (本方案) |
|------|-----------|-----------------|
| 权重 | w = exp(A/λ) > 0 永远为正 | Ã ∈ [-c, c] 可正可负 |
| 差 action 效果 | 仍被"拉向"（力度小） | 被**主动推离** |
| 数学等价 | ≈ reward-weighted regression | ≈ policy gradient (方差更低) |
| 梯度信号 | 单向：只有拉力，无推力 | 双向：好的拉，差的推 |
| 95% 负 advantage | 权重接近 0，几乎不学 | 负权重，主动推离差动作 |

**FPI 失败的根本原因**：在 sparse reward 场景下，95% 的样本 A < 0，exp(A/λ) ≈ 0，这些样本几乎不产生梯度信号。5% 好样本权重很大但数量少，导致有效信号不足，velocity field 退化。

**AWM 的改进**：负 advantage 的样本产生负权重，梯度方向反转，主动推离差 action。这意味着 95% 的"差"样本仍然产生有用的学习信号。

## 4. 时间约定

与 FPI 相同，代码使用 t (t=0→clean, t=1→noise)：

| 概念 | AWM 论文 (τ) | 代码 (t) |
|------|-------------|---------|
| Clean data (action) | τ = 1 | t = 0 |
| Pure noise | τ = 0 | t = 1 |
| 插值 | x_τ = τ·a + (1-τ)·ε | x_t = (1-t)·a + t·ε |
| 目标 velocity | a - ε | ε - a |

## 5. Value Network 架构（与 FPI 相同，π-RL 方案）

```
Observation → VLM Encoder (PaliGemma) → prefix_output
    └──→ [detach] → ValueHead MLP → scalar V(o, l)
```

关键设计：
- V(o, l) 只依赖 observation，**不依赖 action**（action-independent）
- `detach_critic_input: True`：value 梯度不穿透 VLM backbone
- **ForwardType.VALUE**：只做 VLM encoding + value head，跳过 ODE 采样

Value 训练（π-RL 风格）：
- **Progress loss**：`||V(o_t) - (t/T)·R||²`，reward-scaled，所有轨迹都贡献
- **TD loss**：`||V(o_t) - y_t||²`，y_t = r_t + γ·V_target(o_{t+1})
- V_target: EMA 模型的 value head（target network）

## 6. 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `rlinf/algorithms/losses.py` | `compute_flow_awm_loss()` — 可负权重的 AWM MSE |
| `rlinf/algorithms/credit_assignment.py` | `compute_flow_awm_advantages()` — normalize + clip（无 exp） |
| `rlinf/algorithms/registry.py` | `"flow_awm"` 加入 bypass 列表 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | AWM 初始化 + 信用分配 + 训练分支 (含 KL) |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | AWM 检测 + EMA + `_compute_awm_annotations()` |
| `examples/embodiment/config/libero_object_flowawm_openpi_quickstart.yaml` | YAML 配置 |

**复用 FPI 已有的基础设施**（无需修改）：
- `base_policy.py` — `ForwardType.VALUE` 枚举值
- `openpi_action_model.py` — `forward_value()` + VALUE 分发 + `get_value_from_vlm()` detach 支持

## 7. 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `flow_awm_gamma` | 0.99 | TD 折扣因子 |
| `flow_awm_gae_lambda` | 0.95 | GAE 平滑因子 (1.0=MC, 0.0=TD(0)) |
| `flow_awm_adv_clip` | 2.0 | Advantage 对称裁剪阈值 c: Ã ∈ [-c, c] |
| `flow_awm_value_coeff` | 0.5 | Value loss 系数 α |
| `flow_awm_kl_coeff` | 0.1 | Trust region KL 系数 β (AWM 论文默认值) |
| `flow_awm_ema_beta` | 0.995 | EMA target network 更新率 |
| `flow_awm_t_min` | 0.0 | Flow time 采样下界 |
| `flow_awm_t_max` | 1.0 | Flow time 采样上界 |
| `flow_awm_value_warmup` | 10 | Warmup epochs（Ã=0 纯 BC，只训 value） |
| `detach_critic_input` | True | 阻止 value 梯度穿透 VLM |

**调参建议**：
- `adv_clip` (c)：从 2.0 开始。如果推离太猛导致 success rate 下降，调到 1.0；如果改进太保守，调到 3.0
- `kl_coeff` (β)：0.1 是 AWM 论文默认值，比 FPI 的 1.0 小。AWM 有 advantage clipping 作为安全机制，KL 可以更轻
- `value_warmup`：10 是最低限。Value head 需要足够时间学到有意义的 advantage 信号
- 如果 success rate 震荡剧烈 → 增大 `kl_coeff` 或减小 `adv_clip`

## 8. 与其他方法的对比

| 特性 | FlowIPO | FlowSAR | FPI | AWM-VLA |
|------|---------|---------|-----|---------|
| 核心思想 | ODE 插值 + 对比 | 自标注 + 能量加权 | exp(A/λ) 加权 FM | 归一化 A 加权 FM |
| 需要 Value Network | 否 | 否 | **是** | **是** |
| 信用分配 | Policy divergence δ_i | Reconstruction error e_i | exp(A_i/λ) | clip(normalize(A_i)) |
| 权重可为负 | 否 | 否 | 否 | **是** |
| Loss 形式 | Interpolated FM | Energy-weighted FM | w · MSE | Ã · MSE |
| Trust Region | ref_action 插值 | `||v_θ - v_old||²` | `||v_θ - v_old||²` | `||v_θ - v_old||²` |
| (t, ε) 采样 | Rollout 预计算 | Rollout 预计算 | Rollout 预计算 | Rollout 预计算 |
| 理论保证 | 启发式 | 启发式 | Policy improvement | **≈ Policy gradient** |

## 9. 一键启停

```bash
# 训练
bash run_embodiment.sh libero_object_flowawm_openpi_quickstart

# 评估
bash run_embodiment.sh libero_object_flowawm_openpi_quickstart --only_eval

# 恢复训练
bash run_embodiment.sh libero_object_flowawm_openpi_quickstart --resume_dir <path>
```

## 10. 监控指标

TensorBoard 中关注：

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `awm_loss` | AWM 加权 FM loss (可正可负) | 下降 |
| `awm_mse` | 原始 FM MSE（无权重） | 下降 |
| `awm_kl_loss` | Trust region `||v_θ - v_old||²` | 小且稳定 |
| `awm_value_loss` | Value 总 loss | 下降 |
| `awm_td_loss` | TD loss | 下降 |
| `awm_prog_loss` | Progress loss | 下降 |
| `awm_v_mean` | 平均 value 预测 | 上升（随成功率提升） |
| `awm_adv_mean` | 平均 advantage | ≈ 0（归一化后） |
| `awm_adv_std` | Advantage 标准差 | ≈ 1（归一化后） |
| `awm_adv_pos_frac` | 正 advantage 比例 | > 0（成功样本） |
| `awm_adv_neg_frac` | 负 advantage 比例 | > 0（失败样本） |
| `awm/episode_reward` | Episode 平均奖励 | 上升 |
| `awm/success_rate` | 成功率 | 上升 |
| `awm/warmup_remaining` | 剩余 warmup epochs | 递减到 0 |

**训练阶段**：
1. **Warmup (前 N epochs)**：Ã=0（纯 BC），value head 学习基本时间结构。KL loss 应该很小。
2. **过渡期 (N ~ N+10)**：advantage 开始生效，pos_frac 和 neg_frac 分化。
3. **稳定改进期**：正负 advantage 区分好坏 action，success_rate 稳步上升，KL loss 维持稳定。

**异常诊断**：
- `kl_loss` 持续增大 → 策略在漂移，增大 `kl_coeff`
- `awm_loss` 大幅为负且不收敛 → 推离信号太强，减小 `adv_clip`
- `awm_v_mean` 发散 → value 训练不稳定，检查 `detach_critic_input`
- Success rate 下降到 0 → 推离主导，增大 `kl_coeff` 或减小 `adv_clip` 到 1.0

## 11. 关键公式速查

```
Advantage:        A_t = V(o_{t+1}) - V(o_t)
Normalized:       Ã_t = clip((A_t - μ) / (σ + ε), -c, c)

AWM Loss:         L_AWM = Ã · ||v_θ(x_t, t) - (ε - a)||²
Trust Region:     L_KL  = ||v_θ(x_t, t) - v_old(x_t, t)||²
Value Loss:       L_V   = L_TD + L_prog
  TD Target:      y_t = r_t + γ · V_target(o_{t+1})
  Progress:       p_t = (t/T) · R   (R=episode reward)

Total Loss:       L = L_AWM + α · L_V + β · L_KL
```

## 12. AWM 梯度分析

对 θ 求梯度：

```
∇_θ L_AWM = E[Ã · 2(v_θ - u) · ∇_θ v_θ]
```

- Ã > 0：v_θ 被推向 u（学习好 action 的 flow target）
- Ã < 0：v_θ 被推离 u（遗忘差 action 的 flow target）

数学上等价于 policy gradient：
```
∇_θ L_AWM ≈ -E[A_t · ∇_θ log π_θ(a_t | o_t, l)] + O(ε_ELBO)
```

方差比 Flow-SDE+PPO 低，因为在 clean action 上计算（而非 noisy intermediate）。
