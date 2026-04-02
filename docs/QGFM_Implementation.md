# QGFM (Q-Guided Flow Matching) — 实现文档

## 1. 算法概述

QGFM 解决 Flow VLA RL 的三种 gradient carrier 困境——CFM 误差 on-policy vanish、SDE noise 需要复杂系统、Q-gradient 穿 ODE 不稳定。

```
a' = a + η · ∇_a Q(s, a) / ||∇_a Q||     # Q-gradient perturbed action
L_QGFM = ||v_θ(x_t, t | s) - (ε - a')||²  # 标准 FM loss with perturbed target
x_t = (1-t) · a + t · ε                    # 用原始 a 做插值
```

核心 insight：**不让 Q-gradient 穿过 ODE（不做 BPTT），而是用 Q-gradient 扰动 flow matching 的 target action，然后让 flow matching 自己学。**

Gradient carrier 分析：
```
标准 FM:  ∇_θ L ∝ (v_θ - u) · ∇_θ v_θ           → on-policy: (v_θ - u) ≈ 0, gradient vanish
QGFM:    ∇_θ L ∝ (v_θ - u') · ∇_θ v_θ
       = [(v_θ - u) + η·ĝ] · ∇_θ v_θ
       ≈ η · ĝ · ∇_θ v_θ                       → 始终非零 (η > 0, ĝ ≠ 0)
```

## 2. 数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Rollout Worker                                │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ π_θ rollout — 标准 ODE 采样（纯 ODE，无 SDE）              │           │
│  │                                                            │           │
│  │  特点：                                                     │           │
│  │  - 标准 ODE 去噪                                           │           │
│  │  - 收集 vlm_embedding = pool(prefix_output).detach()       │           │
│  │  - 无 EMA 模型，无 SDE noise，无 NFT snapshots             │           │
│  └──────────────────────────────────────────────────────────┘           │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ Rollout 数据:                                                │       │
│  │  actions       = ODE 采样的动作 [n_steps, batch, chunk, dim] │       │
│  │  vlm_embedding = pooled VLM features [n_steps, batch, hidden]│       │
│  │  rewards       = 环境反馈 [n_steps, batch, chunk]           │       │
│  └──────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  send to actor
┌──────────────────────────────────────────────────────────────────────────┐
│                           Actor Worker                                   │
│                                                                          │
│  ┌─── Credit Assignment ───────────────────────────────────────────┐    │
│  │  构建 SARSA 转移对:                                              │    │
│  │    (vlm_emb_t, a_t, r_t, vlm_emb_{t+1}, a_{t+1})              │    │
│  │  Q-network 通过 TD bootstrap 提供 per-step credit               │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│        │                                                                │
│        ▼                                                                │
│  ┌─── Training (per mini-batch) ─────────────────────────────────┐    │
│  │                                                                 │    │
│  │  === Critic Update (每步) ===                                   │    │
│  │    y = r + γ · min_j Q̄_j(s', a')   [target Q]                │    │
│  │    L_Q = ½ ||Q_j(s, a) - y||²       [双Q, MSE]                │    │
│  │                                                                 │    │
│  │  === Actor Update (η > 0 时, after Q warm-up) ===              │    │
│  │    1. Compute Q-gradient: ĝ = ∇_a min_j Q_j / ||...||         │    │
│  │    2. Perturb target: a' = a + η · ĝ                           │    │
│  │    3. Sample fresh: t ~ U[0,1], ε ~ N(0,I)                     │    │
│  │    4. x_t = (1-t)·a + t·ε  (原始 a 做插值)                     │    │
│  │    5. v_θ = model(VELOCITY, x_t, t)                             │    │
│  │    6. L_QGFM = ||v_θ - (ε - a')||²                             │    │
│  │                                                                 │    │
│  │  === Target Q Update ===                                        │    │
│  │    Q̄ ← (1-τ)Q̄ + τQ                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. QGFM vs RK-FAC vs FlowIPO: 关键差异

| 特性 | FlowIPO | RK-FAC | **QGFM** |
|------|---------|--------|----------|
| Forward type | VELOCITY (1 suffix) | RKFAC_ODE (2K suffix) | **VELOCITY (1 suffix)** |
| Gradient carrier | w·(v_θ-u) | Q-grad through ODE | **η·∇_a Q (perturbed target)** |
| 需要 ODE loop? | 否 | 是 (K步 Euler) | **否** |
| 需要 weight swap? | 否 | 是 (frozen expert) | **否** |
| 需要 gradient checkpointing? | 否 | 是 | **否** |
| 内存需求 | 低 | 高 (80GB A100) | **低 (同 FlowIPO)** |
| Trust region | Advantage weighting | E_res (残差动能) | **η schedule (小步保守)** |
| Critic | 无 (或 V-head) | Q(s,a) frozen-VLM | **Q(s,a) frozen-VLM** |
| Actor loss | Weighted FM MSE | α·E_res - Q(s,a_θ) | **标准 FM MSE** |
| Off-policy potential | Partial | Semi-on-policy | **Semi-on-policy (可扩展 replay)** |

## 4. 修改的文件一览

| 文件 | 修改内容 | ~Lines |
|------|---------|--------|
| `rlinf/algorithms/registry.py` | `"flow_qgfm"` 加入 bypass 列表 | +1 |
| `rlinf/algorithms/losses.py` | `@register_policy_loss("flow_qgfm")` placeholder | +20 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | `_is_flow_qgfm` + `collect_vlm_embedding` | +3 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | Init (Q, target Q) + credit assignment + training branch | +120 |
| `examples/embodiment/config/libero_object_qgfm_openpi_quickstart.yaml` | YAML 配置 | 新文件 |

**不需要修改**：
- `base_policy.py` — 无新 ForwardType
- `openpi_action_model.py` — 无新 forward 方法（复用 VELOCITY）

## 5. 关键实现细节

### 5.1 Q-Gradient Perturbation

计算 ∇_a Q 的过程：
```python
# actions: [batch, chunk, dim] from rollout
a_for_grad = actions[:, :chunk, :dim].detach().clone().requires_grad_(True)
a_flat = a_for_grad.reshape(batch, -1).float()

# Forward through Q-network
q_vals = Q_network(vlm_emb.detach().float(), a_flat)  # [batch, 2]
q_min = q_vals.min(dim=-1).values.sum()

# Backward to get Q-gradient w.r.t. actions
q_min.backward()
q_grad = a_for_grad.grad  # [batch, chunk, dim]

# Normalize to unit direction
q_grad_hat = q_grad / (||q_grad|| + ε)

# Perturb target
a_prime = a + η · q_grad_hat  # detach before FM loss
```

**关键点**：`a_prime` 必须 detach — Q-gradient 只扰动 target 方向，不让 FM loss 回传到 Q-network。

### 5.2 η Schedule (Perturbation Step Size)

```
Phase 1 (step ≤ q_warmup_iters):  η = 0  (纯 BC, 只训 Q)
Phase 2 (ramp up):                η = η_init + (η_max - η_init) · progress
Phase 3 (steady state):          η = η_max
```

设计理由：
- 早期 Q 不准 → η=0 避免错误方向扰动 → 退化为 BC → 不会比 SFT 差
- 后期 Q 学到 value → 加大 η → 更 aggressive improvement
- η_max 控制最大偏离 → 保守 (0.01-0.05 for LIBERO action scale)

### 5.3 Fresh (t, ε) Sampling

QGFM 在训练时 sample 新的 `t ~ U[0,1]` 和 `ε ~ N(0,I)`（不用 rollout 时预算的）：
- 更灵活：支持多 epoch 训练（每 epoch 不同的 t, ε 降低 variance）
- 无需修改 rollout worker 的 forward_inputs 存储

### 5.4 Interpolation 点选择

使用 **Choice 1**（§3.2.5 of Research Proposal）：
```
x_t = (1-t) · a + t · ε    # 用原始 a 做插值
u' = ε - a'                 # perturbed target velocity
```

**不**改变 interpolation 点（不用 a' 做插值），只改 target。理由：
- rollout 时不需要 a'
- η 小时两种 choice 差异 O(τη) 可忽略
- 更简单

### 5.5 与 loss_mask 的兼容

当 `loss_mask` 存在时，QGFM loss 正确地按 mask 加权：
```python
per_sample_loss = (diff ** 2).sum(dim=-1).mean(dim=-1)  # [batch]
if loss_mask is not None:
    lm = loss_mask.view(-1) if loss_mask.dim() > 1 else loss_mask
    loss = (per_sample_loss * lm).sum() / lm.sum().clamp(min=1)
else:
    loss = per_sample_loss.mean()
```

**注意**：`loss_mask` 在 `chunk_level` reward 模式下形状为 `[batch, 1]`，需要 `view(-1)` 展平到 `[batch]` 以与 `per_sample_loss` 匹配。

### 5.7 Safety Guards

- **`has_transitions` guard**: Actor update 仅在 `has_transitions=True` 时执行，防止对空 transition batch 计算 Q-gradient
- **VLM embedding None**: 当 `forward_inputs` 中无 `vlm_embedding` 时，打印 warning 并跳过 actor update（loss=0）

### 5.6 Transition Data Padding

同 RK-FAC：transition 对是 `(s_t, a_t, r_t, s_{t+1}, a_{t+1})`，只有 `n_steps - 1` 个。Zero-pad 到 `n_steps` + `qgfm_valid` mask。

## 6. 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `qgfm_eta_init` | 0.001 | 初始扰动步长 (小 → 近 BC) |
| `qgfm_eta_max` | 0.02 | 最大扰动步长 |
| `qgfm_eta_warmup_iters` | 50 | η 从 init 线性增长到 max 的步数 |
| `qgfm_q_warmup_iters` | 20 | 纯 BC（η=0）步数，等 Q 收敛 |
| `qgfm_gamma` | 0.99 | TD 折扣因子 |
| `qgfm_tau` | 0.005 | Target Q-network EMA rate |
| `qgfm_q_lr` | 1e-4 | Q-network 学习率 |
| `qgfm_q_hidden_dims` | [512, 256, 128] | Q MLP 隐藏层维度 |
| `qgfm_actor_delay` | 1 | Actor 每 N 步 critic 更新后更新（默认 1 = 每步）|
| `qgfm_grad_clip` | 1.0 | Q-network 梯度裁剪 |
| `qgfm_norm_q_grad` | True | 归一化 ∇_a Q 到单位方向 |
| `qgfm_vlm_hidden` | 2048 | VLM 嵌入维度 (PaliGemma 2B) |

**调参建议**：
- `qgfm_eta_max`: 最关键参数。太小 → 接近 BC，改善慢。太大 → a' 偏离 policy 支撑集，FM 拟合不稳定。从 0.02 开始，观察 `qgfm_loss` 下降速度
- `qgfm_q_warmup_iters`: Q TD loss 开始下降后再启用 perturbation。观察 `actor/q_loss`
- `qgfm_norm_q_grad`: True = 方向信息 (稳定); False = 方向+幅度 (可能更快但不稳定)
- `update_epoch`: QGFM 用标准 BC loss，安全做多 epoch。4 是好起点

## 7. 监控指标

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `actor/q_loss` | Q-network TD loss | 逐步下降 |
| `actor/qgfm_loss` | QGFM flow matching loss | 下降（policy 学习 perturbed target）|
| `actor/eta` | 当前扰动步长 η | 从 0 线性增长到 eta_max |
| `actor/q_mean` | Q 值均值 | 从 0 增长（学到正值 Q for success）|
| `actor/td_target_mean` | TD target 均值 | 稳定增长 |
| `actor/q_warmup` | 是否在 Q warm-up 阶段 | 1→0（warm-up 结束后）|
| `qgfm/success_rate` | 成功率 | 核心指标，稳步上升 |
| `qgfm/episode_reward_mean` | Episode 奖励均值 | 上升 |

**训练阶段**：
1. **Q warm-up (step 0-20)**: η=0, 纯 BC, Q loss 下降
2. **η ramp-up (step 20-70)**: η 线性增长, QGFM loss 可能先升后降
3. **稳定训练 (step 70+)**: η=η_max, Q 和 policy 协同改进, success_rate 上升

## 8. 一键启停

```bash
# 训练
bash run_embodiment.sh libero_object_qgfm_openpi_quickstart

# 评估
bash run_embodiment.sh libero_object_qgfm_openpi_quickstart --only_eval

# 恢复训练
bash run_embodiment.sh libero_object_qgfm_openpi_quickstart --resume_dir <path>
```

## 9. 关键公式速查

```
Q-gradient:       ĝ = ∇_a min_j Q_j(s, a) / ||∇_a min_j Q_j(s, a)||
Perturbation:     a' = a + η · ĝ
Flow matching:    x_t = (1-t)·a + t·ε,  u' = ε - a'
QGFM loss:       L = ||v_θ(x_t, t | s) - u'||²
Effective grad:   ∇_θ L ≈ -2η · ĝ · ∇_θ v_θ   (non-vanishing!)

Critic loss:      y = r + γ · min_j Q̄_j(s', a')
                  L_Q = ½ Σ_j ||Q_j(s, a) - y||²

Target update:    Q̄ ← (1-τ)Q̄ + τQ

η schedule:       η = 0                               (step ≤ q_warmup)
                  η = η_init + (η_max - η_init)·p     (ramp up)
                  η = η_max                            (steady state)
```
