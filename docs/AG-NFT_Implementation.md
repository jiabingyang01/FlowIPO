# AG-NFT (Advantage-Guided NFT) — 实现文档

## 1. 算法概述

AG-NFT 的核心思想：**给 NFT 加一个 value network，用 GAE advantage 的连续值作为每步标签 y ∈ [-1, +1]**。

```
L_total = L_NFT(y_t, ΔE_t) + α · L_V
          ↑ 对比学习 loss（y 可连续）  ↑ value 训练
```

关键创新：**NFT 的 softplus 结构天然就是自校准的门控机制**：

```
y_t = normalized_advantage(A_t)     # ∈ [-1, +1]，连续值
logit = (dpo_beta / 2) · y_t · ΔE
L = softplus(logit)
∂L/∂θ ∝ y_t                        # 当 |y| ≈ 0 时，梯度 → 0
```

- Value network 刚初始化时不准确 → GAE advantage 是噪声 → |y| ≈ 0 → NFT **自动忽略**
- Value network 训练好后 → advantage 有意义 → |y| 增大 → 每步信用分配**自动激活**
- **不需要任何显式的 warmup 或 phase 切换**

架构级创新：**解耦 critic 的两个角色**：
- Critic 提供**方向**：sign(A_t) 决定哪些步好、哪些步差
- NFT 提供**幅度**：softplus 结构保证 O(1) 恒定梯度
- 两者互不干扰，各做各的

## 2. 数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Rollout Worker                                │
│  ┌──────────────┐                                                       │
│  │ π_θ rollout   │──→ (o_t, a_t, r_t) per step                        │
│  │ (SDE chain)   │──→ NFT snapshots: (x_t, v_t, x_next, step_idx)     │
│  └──────────────┘                                                       │
│        │                                                                 │
│        ├→ VLM prefix_output → [detach] → ValueHead → V(o_t, l)         │
│        │   (零额外 VLM forward 成本，复用已有的 prefix encoding)         │
│        │                                                                 │
│        └→ forward_inputs + prev_values                                  │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  send to actor
┌──────────────────────────────────────────────────────────────────────────┐
│                           Actor Worker                                   │
│                                                                          │
│  ┌─── Credit Assignment (AG-NFT GAE) ────────────────────────────┐     ��
│  │  step_rewards = rewards.sum(dim=-1)        # [n_steps, batch]  │     │
│  │  GAE: A_t, R_t = GAE(step_rewards, V, dones, γ, λ)            │     │
│  │  advantages = clamp(A_t, -c, c)            # → y ∈ [-1, +1]   │     │
│  │  agnft_returns = R_t                        # value 训练目标    │     │
│  │                                                                 │     │
│  │  自校准: |y| ≈ 0 → softplus 梯度 ≈ 0 → 自动忽略噪声标签       │     │
│  └────────────────────────────────────────────────────────────────┘     │
│        │                                                                │
│        ▼                                                                │
│  ���─── Training (per mini-batch, SINGLE VLM forward) ─────────────┐    │
│  │                                                                 │    │
│  │  ForwardType.VELOCITY(return_value=True):                       │    │
│  │    VLM prefix → [一次 VLM forward]                              │    │
│  │    ├→ suffix → velocity head → v_θ     (有梯度, 训 policy)      │    │
│  │    └→ [detach] → value head → V(o_t)   (无梯度穿透 VLM)        │    │
│  │                                                                 │    │
│  │  NFT Loss (不变):                                                │    │
│  │    y = map(advantages, [-c,c] → [-1,+1])                        │    │
│  │    ΔE = Mahalanobis(v_pos, v_neg, x_t, x_next, schedule)       │    │
│  │    L_NFT = softplus((dpo_β/2) · y · ΔE) + kl_β · ||v_θ-v_old||² │  │
│  │                                                                 │    │
│  │  Value Loss (新增):                                              │    │
│  │    L_V = ||V_θ(o_t) - R_t||²                                   │    │
│  │                                                                 │    │
│  │  Total: L = L_NFT + α · L_V                                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. AG-NFT vs 其他方法的对比

| 特性 | NFT (terminal binary) | AG-NFT (本方案) | PPO | AWM |
|------|-----------------------|-----------------|-----|-----|
| 信用分配 | Episode 级 ±1 | Step 级 GAE | Step 级 GAE | Step 级 GAE |
| 梯度幅度 | O(1) 恒定 | O(1) 恒定 | 高方差 | → 0 消失 |
| 需要 Value Network | 否 | **是** | 是 | 是 |
| 长视野支持 | 差（同 label） | **好** | 好 | 好 |
| 自校准 | N/A | **是** | 否 | 否 |
| 理论保证 | DPO contrastive | DPO contrastive | Policy gradient | ≈ PG |

**AG-NFT 解决 NFT 的唯一弱点**（长视野信用分配），同时保留 NFT 的核心优势（O(1) 梯度，无消失问题）。

## 4. 自校准机制详解

NFT loss 对 advantage label y 的灵敏度：

```
L = softplus((dpo_β/2) · y · ΔE)
∂L/∂θ = σ(logit) · (dpo_β/2) · y · ∂ΔE/∂θ
```

| y 值 | 含义 | 梯度效果 |
|------|------|---------|
| y = +1 | 强正 advantage（好步） | 完整正梯度 → 学习好动作 |
| y = -1 | 强负 advantage（差步） | 完整负梯度 → 推离差动作 |
| y ≈ 0 | 不确定（value 不准） | 梯度 ≈ 0 → **自动忽略** |
| y = +0.3 | 弱正 advantage | 30% 梯度 → 弱学习信号 |

训练阶段：
1. **初期**（value 不准）：大部分 y ≈ 0 → NFT 行为接近无标签 → 安全
2. **中期**（value 逐渐准确）：|y| 逐渐增大 → 每步信用分配逐渐激活
3. **后期**（value 准确）：y ≈ ±1 → 等价于有 per-step 标签的 NFT

诊断指标：
- `agnft/y_near_zero_frac`：|y| < 0.1 的比例 → 初期 ≈ 1.0，逐渐下降
- `agnft/y_strong_frac`：|y| > 0.5 的比例 → 初期 ≈ 0，逐渐上升

## 5. Value Network 架构（复用 FPI/AWM 基础设施）

```
Observation → VLM Encoder (PaliGemma) → prefix_output
    └──→ [detach] → ValueHead MLP (512→256→128→1) → scalar V(o, l)
```

关键设计：
- V(o, l) 只依赖 observation，**不依赖 action**
- `detach_critic_input: True`：value 梯度不穿透 VLM backbone
- `return_value=True`：一次 VLM forward 同时出 velocity + value（零额外成本）

## 6. 修改的文件

| 文件 | 修改内容 | 状态 |
|------|---------| ---- |
| `rlinf/workers/actor/fsdp_actor_worker.py` | `_flow_nft_cfg` 新增 3 个 GAE 参数 | ✅ |
| `rlinf/workers/actor/fsdp_actor_worker.py` | `_compute_flow_nft_credit_assignment()` 新增 `"gae"` 分支 | ✅ |
| `rlinf/workers/actor/fsdp_actor_worker.py` | NFT 训练分支新增 `return_value` + value loss | ✅ |
| `examples/embodiment/config/libero_object_agnft_openpi_quickstart.yaml` | AG-NFT 配置 | ✅ |

**复用已有的基础设施**（无需修改）：

| 组件 | 位置 | 说明 |
|------|------|------|
| Value head 创建 | `openpi_action_model.py:139-154` | `add_value_head: True` 自动创建 |
| Rollout 中计算 value | `openpi_action_model.py:498-500, 534-535` | `use_vlm_value=True` 自动返回 |
| `return_value=True` | `openpi_action_model.py:794, 846-850` | AWM 已在用 |
| GAE 计算 | `advantages.py:24-86` | 已注册，直接调用 |
| Value head 优化器/LR | `fsdp_model_manager.py:417-486` | `value_lr` 已支持 |
| Rollout 收集 prev_values | `huggingface_worker.py:354, 398` | 已自动收集 |

## 7. 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `flow_nft_adv_type` | `"gae"` | AG-NFT 核心开关（`"terminal_binary"` 回退到标准 NFT） |
| `flow_nft_value_coeff` | 0.5 | Value loss 系数 α |
| `gamma` | 0.99 | GAE 折扣因子 |
| `gae_lambda` | 0.95 | GAE 平滑因子 |
| `flow_nft_adv_clip_max` | 1.0 | Advantage 裁剪，映射到 y ∈ [-1, +1] |
| `add_value_head` | True | 启用 VLM value head |
| `value_after_vlm` | True | V(o,l) 从 VLM prefix 特征计算 |
| `detach_critic_input` | True | 阻止 value 梯度穿透 VLM |
| `value_lr` | 1e-4 | Value head 单独学习率 |
| `critic_warmup_steps` | 0 | 自校准使 warmup 非必需 |

**调参建议**：
- `flow_nft_value_coeff`：从 0.5 开始。如果 value 训练慢 → 增大到 1.0
- `flow_nft_adv_clip_max`：从 1.0 开始。如果信用分配太激进 → 减小到 0.5
- 如果初期 success rate 下降 → 检查 `y_near_zero_frac` 是否接近 1.0（应该是）

## 8. 一键启停

```bash
# 训练
bash run_embodiment.sh libero_object_agnft_openpi_quickstart

# 评估
bash run_embodiment.sh libero_object_agnft_openpi_quickstart --only_eval

# 恢复训练
bash run_embodiment.sh libero_object_agnft_openpi_quickstart --resume_dir <path>

# A/B 测试：切换回标准 NFT（只改 adv_type）
bash run_embodiment.sh libero_object_agnft_openpi_quickstart algorithm.flow_nft_adv_type=terminal_binary
```

## 9. 监控指标

TensorBoard 中关注：

| 指标 | 含义 | 期望趋势 |
|------|------|---------:|
| `agnft/y_near_zero_frac` | |y| < 0.1 的比例（自校准诊断） | 从 ~1.0 下降 |
| `agnft/y_strong_frac` | |y| > 0.5 的比例（信用分配激活度） | 从 ~0 上升 |
| `actor/agnft_value_loss` | Value MSE loss | 下降 |
| `actor/agnft_v_mean` | 平均 value 预测 | 上升 |
| `agnft/raw_gae_adv_mean` | 原始 GAE advantage 均值 | ≈ 0 |
| `agnft/raw_gae_adv_std` | 原始 GAE advantage 标准差 | 逐渐增大 |
| `nft/success_rate` | 成功率 | 上升 |
| `nft/episode_reward` | Episode 奖励 | 上升 |
| `actor/nft_loss` | NFT contrastive loss | 下降 |
| `actor/kl_loss` | Trust region ||v_θ - v_old||² | 小且稳定 |

**训练阶段**：
1. **自校准期**（初期）：`y_near_zero_frac ≈ 1.0`，NFT 接近无标签模式，value 训练中
2. **激活期**：`y_strong_frac` 上升，per-step 信用分配开始生效
3. **稳定改进期**：per-step 标签区分好坏步，success_rate 稳步上升

**异常诊断**：
- `y_near_zero_frac` 始终接近 1.0 → value 没学到东西，检查 `value_lr`
- `kl_loss` 持续增大 → 策略漂移，增大 `flow_nft_kl_beta`
- Success rate 突然下降 → per-step 标签噪声太大，增大 `flow_nft_adv_clip_max`（限制 y 范围）

## 10. 关键公式速查

```
GAE Advantage:    A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
                  δ_t = r_t + γ · V(o_{t+1}) - V(o_t)

Label Mapping:    y_t = clamp(A_t, -c, c) → normalize to [-1, +1]

NFT Loss:         ΔE = Mahalanobis(v_pos, v_neg, x_t, x_next, σ_t)
                  L_NFT = softplus((dpo_β/2) · y_t · ΔE) + kl_β · ||v_θ - v_old||²

Value Loss:       L_V = ||V_θ(o_t) - R_t||²
                  R_t = GAE returns

Total Loss:       L = L_NFT + α · L_V
```
