# FPI (Flow Policy Iteration) — 实现文档

## 1. 算法概述

FPI 的核心思想：**Policy Improvement = Advantage-Weighted Flow Matching + Trust Region**。

给标准 flow matching loss 加一个 per-sample weight `w = exp(A/λ)`，同时用 KL 正则化防止多轮更新中的策略漂移：

```
L_total = w · ||v_θ - u||²  +  β_kl · ||v_θ - v_old||²  +  β_v · L_V
           ↑ advantage-weighted FM    ↑ trust region           ↑ value training
```

- A > 0（好的 action chunk）→ w > 1 → 放大 loss → 更努力学习
- A < 0（差的 action chunk）→ w < 1 → 缩小 loss → 少关注
- A ≈ 0（平均水平）→ w ≈ 1 → 等价于标准 BC

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
│  │ EMA model (_compute_fpi_annotations)                         │       │
│  │   Phase 1: V_target(o_t) via EMA value head → TD bootstrap  │       │
│  │   Phase 2: pre-compute (t, ε, v_old) → trust region anchor  │       │
│  └──────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  send to actor
┌──────────────────────────────────────────────────────────────────────────┐
│                           Actor Worker                                   │
│                                                                          │
│  ┌─── Credit Assignment ───────────────────────────────────────────┐    │
│  │  A_t = V(o_{t+1}) - V(o_t)            (advantage from prev_values)│  │
│  │  A_t ← normalize(A_t)                  (zero-mean unit-var)      │  │
│  │  w_t = exp(A_t/λ) / mean(exp(A_j/λ))   (self-normalized weights)│  │
│  │  td_target = r_t + γ · V_target(o_{t+1})                        │  │
│  │  progress_target = (t/T) · R            (reward-scaled, π-RL)    │  │
│  │  [warmup: w=1 for first N epochs]                                │  │
│  └──────────────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        ▼                                                                │
│  ┌─── Training (per mini-batch) ──────────────────────────────────┐    │
│  │                                                                 │    │
│  │  Phase 1: Value Training                                        │    │
│  │    ForwardType.VALUE → V_θ(o_t)  [detach VLM features]         │    │
│  │    L_TD = ||V_θ(o_t) - td_target||²                            │    │
│  │    L_prog = ||V_θ(o_t) - progress_target||²  (all trajectories)│    │
│  │    L_V = L_TD + L_prog                                          │    │
│  │                                                                 │    │
│  │  Phase 2: Policy Training (pre-computed t, ε from rollout)      │    │
│  │    x_t = (1-t)·a + t·ε    (same t, ε across update epochs)     │    │
│  │    ForwardType.VELOCITY → v_θ(x_t, t | o, l)                   │    │
│  │    u = ε - a  (flow matching target)                            │    │
│  │    L_FPI = w · ||v_θ - u||²                                     │    │
│  │    L_KL  = ||v_θ - v_old||²   (trust region)                   │    │
│  │                                                                 │    │
│  │  Phase 3: Combined                                              │    │
│  │    L = L_FPI + β_v · L_V + β_kl · L_KL                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. 与理论的偏差及其原因

FPI 论文描述的是理想的单步更新。实际工程中需要额外机制保证稳定性：

| 论文理论 | 实践问题 | 工程解决方案 |
|---------|---------|------------|
| 单步 policy improvement | 实际做 4 个 update epoch → 策略漂移 | Trust region: `||v_θ - v_old||²` KL penalty |
| Demo 数据预训练 value | 没有独立的 demo 数据加载器 | Reward-scaled progress: success→t/T, failure→0 |
| 新采样 (t, ε) | 每 epoch 不同的 (t, ε) → v_old 无法做锚 | 预计算 (t, ε, v_old)，4 个 epoch 共用 |
| V(o,l) 与 policy 联合训练 | Value 梯度穿透 VLM → 特征腐蚀 | `detach_critic_input: True` |
| Advantage 直接 exp() | 随 V 改善 advantage 方差增大 → 权重爆炸 | Advantage 标准化 + weight clipping |

## 4. 时间约定

FPI 论文使用 τ（τ=0→noise，τ=1→clean action），代码库使用 t（t=0→clean，t=1→noise）：

| 概念 | FPI 论文 (τ) | 代码 (t) |
|------|-------------|---------|
| Clean data (action) | τ = 1 | t = 0 |
| Pure noise | τ = 0 | t = 1 |
| 插值 | x_τ = τ·a + (1-τ)·ε | x_t = (1-t)·a + t·ε |
| 目标 velocity | a - ε | ε - a |

## 5. Value Network 架构（π-RL 方案）

FPI 复用 OpenPI 模型内置的 value head（`value_after_vlm` 模式）：

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
  - 成功轨迹 (R=1): target = t/T（进度线性增长）
  - 失败轨迹 (R=0): target = 0（value 应该低）
- **TD loss**：`||V(o_t) - y_t||²`，y_t = r_t + γ·V_target(o_{t+1})
- V_target: EMA 模型的 value head（target network）

## 6. 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `rlinf/models/embodiment/base_policy.py` | `ForwardType.VALUE` 枚举值 |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | `forward_value()` + VALUE 分发 + `get_value_from_vlm()` detach 支持 |
| `rlinf/algorithms/losses.py` | `compute_flow_fpi_loss()` — 加权 MSE |
| `rlinf/algorithms/credit_assignment.py` | `compute_flow_fpi_weights()` — 自归一化指数权重 + advantage 标准化 |
| `rlinf/algorithms/registry.py` | `"flow_fpi"` 加入 bypass 列表 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | FPI 初始化 + 信用分配 + warmup + 训练分支 (含 KL) |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | FPI 检测 + EMA + `_compute_fpi_annotations()` (V_target + t,ε,v_old) |

## 7. 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `flow_fpi_lambda` | 5.0 | Advantage 温度。大→BC，小→贪心。5.0 保守 |
| `flow_fpi_gamma` | 0.99 | TD 折扣因子 |
| `flow_fpi_w_min` | 0.0 | 权重裁剪下界 |
| `flow_fpi_w_max` | 5.0 | 权重裁剪上界 |
| `flow_fpi_value_coeff` | 0.5 | Value loss 系数 β_v |
| `flow_fpi_kl_coeff` | 1.0 | Trust region KL 系数 β_kl |
| `flow_fpi_ema_beta` | 0.995 | EMA target network 更新率 |
| `flow_fpi_t_min` | 0.0 | Flow time 采样下界 |
| `flow_fpi_t_max` | 1.0 | Flow time 采样上界 |
| `flow_fpi_value_warmup` | 10 | Warmup epochs（w=1 纯 BC，只训 value） |
| `flow_fpi_adv_normalize` | True | Advantage 标准化防止权重爆炸 |
| `detach_critic_input` | True | 阻止 value 梯度穿透 VLM |

**调参建议**：
- `lambda`：从 5.0 开始。如果改进太慢，调到 2.0；如果不稳定，调到 10.0
- `kl_coeff`：trust region 强度。1.0 通常够用。如果还 drift，加到 2.0
- `value_warmup`：10 是最低限。如果 value 学得慢（低成功率），增加到 20-30
- `w_max`：5.0 防止极端权重。如果 advantage_std 很大，可以降到 3.0

## 8. 与 FlowIPO / FlowSAR 的对比

| 特性 | FlowIPO | FlowSAR | FPI |
|------|---------|---------|-----|
| 核心思想 | ODE 插值 + 对比 | 自标注 + 能量加权 | Advantage 加权 FM |
| 需要 Value Network | 否 | 否 | **是** |
| 信用分配 | Policy divergence δ_i | Reconstruction error e_i | Advantage A_i |
| Loss 形式 | Interpolated FM | Energy-weighted FM | Advantage-weighted FM |
| Trust Region | ref_action 插值 | `||v_θ - v_old||²` | `||v_θ - v_old||²` (同 FlowSAR) |
| (t, ε) 采样 | Rollout 预计算 | Rollout 预计算 | Rollout 预计算 |
| 理论保证 | 启发式 | 启发式 | **Policy improvement** |

## 9. 一键启停

```bash
# 训练
bash run_embodiment.sh libero_object_flowfpi_openpi_quickstart

# 评估
bash run_embodiment.sh libero_object_flowfpi_openpi_quickstart --only_eval

# 恢复训练
bash run_embodiment.sh libero_object_flowfpi_openpi_quickstart --resume_dir <path>
```

## 10. 监控指标

TensorBoard 中关注：

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `fpi_loss` | 加权 FM loss | 下降 |
| `fpi_mse` | 原始 FM MSE（无权重） | 下降 |
| `fpi_kl_loss` | Trust region `||v_θ - v_old||²` | 小且稳定 |
| `fpi_value_loss` | Value 总 loss | 下降 |
| `fpi_td_loss` | TD loss | 下降 |
| `fpi_prog_loss` | Progress loss | 下降 |
| `fpi_v_mean` | 平均 value 预测 | 上升（随成功率提升） |
| `fpi_weight_mean` | 平均权重 | ≈ 1（自归一化） |
| `fpi_weight_std` | 权重标准差 | 适度增长（区分好坏） |
| `fpi_weight_max` | 最大权重 | < w_max |
| `fpi_advantage_mean` | 平均 advantage | ≈ 0 |
| `fpi_advantage_std` | Advantage 标准差 | 适度增长 |
| `fpi_episode_reward` | Episode 平均奖励 | 上升 |
| `fpi_success_rate` | 成功率 | 上升 |
| `fpi/warmup_remaining` | 剩余 warmup epochs | 递减到 0 |

**训练阶段**：
1. **Warmup (前 N epochs)**：w=1（纯 BC），value head 学习基本时间结构。KL loss 应该很小。
2. **过渡期 (N ~ N+10)**：advantage weights 逐渐生效，weight_std 开始增长。
3. **稳定改进期**：weights 分化好坏 action，success_rate 稳步上升，KL loss 维持稳定。

**异常诊断**：
- `kl_loss` 持续增大 → 策略在漂移，增大 `kl_coeff`
- `weight_max` 频繁触顶 → 减小 `lambda` 或 `w_max`
- `fpi_v_mean` 发散 → value 训练不稳定，检查 `detach_critic_input`

## 11. 关键公式速查

```
Advantage:        A_t = V(o_{t+1}) - V(o_t)
Normalized Adv:   Â_t = (A_t - mean) / std
Weight:           w_t = exp(Â_t/λ) / mean_j(exp(Â_j/λ)),  clamp [w_min, w_max]

FPI Loss:         L_FPI = w · ||v_θ(x_t, t) - (ε - a)||²
Trust Region:     L_KL  = ||v_θ(x_t, t) - v_old(x_t, t)||²
Value Loss:       L_V   = L_TD + L_prog
  TD Target:      y_t = r_t + γ · V_target(o_{t+1})
  Progress:       p_t = (t/T) · R   (R=episode reward)

Total Loss:       L = L_FPI + β_v · L_V + β_kl · L_KL
```
