# RK-FAC (Residual-Kinetic Flow Actor-Critic) — 实现文档

## 1. 算法概述

RK-FAC 解决 Flow VLA RL 的"不可能三角"——**Temporal Credit（长任务）、OOD Generalization、Sample Efficiency** 无法同时满足。

```
L_actor = α · E_res - min_j Q_j(s, a_θ)
          ↑ 残差动能信任域   ↑ Q梯度穿过可微ODE
```

核心创新：

1. **残差动能 E_res** = ∫ ½||u_θ - u_pre||² dτ 作为 likelihood-free KL bound
   - Girsanov 定理: D_KL(P_θ || P_pre) = E_res / σ²（精确，非上界）
   - 替代 PPO clipping 和 NFT ad-hoc L2 trust region
2. **Frozen-VLM Q**: Q_φ(sg[h_VLM(o)], a) — Q 梯度不回传到 VLM → OOD robust
3. **Q-gradient through ODE**: 策略改进通过 Q 梯度穿过可微 Euler ODE → step-level temporal credit

## 2. 数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Rollout Worker                                │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ π_θ rollout — 标准 ODE 采样                                │           │
│  │                                                            │           │
│  │  特点：                                                     │           │
│  │  - 标准 ODE 去噪（不需要 SDE snapshot）                     │           │
│  │  - 收集 vlm_embedding = pool(prefix_output).detach()       │           │
│  │  - 无 EMA 模型（frozen reference 在 actor worker 维护）     │           │
│  └──────────────────────────────────────────────────────────┘           │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ Rollout 数据:                                                │       │
│  │  actions      = ODE 采样的动作 [n_steps, batch, chunk, dim]  │       │
│  │  vlm_embedding = pooled VLM features [n_steps, batch, hidden]│       │
│  │  rewards       = 环境反馈 [n_steps, batch, chunk]           │       │
│  │  dones         = 终止标志                                    │       │
│  │                                                              │       │
│  │  NO NFT snapshots (nft_xt, nft_v, etc.)                     │       │
│  │  NO EMA reference model                                      │       │
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
│  │  Q-network 通过 TD bootstrap 自动提供 per-step credit           │    │
│  │  不需要传统 advantage 计算                                       │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│        │                                                                │
│        ▼                                                                │
│  ┌─── Training (per mini-batch) ─────────────────────────────────┐    │
│  │                                                                 │    │
│  │  === Critic Update (每步) ===                                   │    │
│  │    y = r + γ · min_j Q̄_j(s', a')   [target Q, 存储的 a']     │    │
│  │    L_Q = ½ ||Q_j(s, a) - y||²       [双Q, MSE]                │    │
│  │                                                                 │    │
│  │  === Actor Update (每N步) ===                                   │    │
│  │    ForwardType.RKFAC_ODE:                                       │    │
│  │      1 VLM prefix → KV cache  [一次 VLM forward]               │    │
│  │      K suffix forwards (Euler ODE): x → a_θ                    │    │
│  │      K frozen suffix forwards: u_pre (weight swap)              │    │
│  │      E_res = Σ_k ½||u_θ_k - u_pre_k||² · dt                   │    │
│  │                                                                 │    │
│  │    L_actor = α · E_res - min_j Q_j(vlm_emb, a_θ)              │    │
│  │    梯度穿过 ODE: ∇_θ Q(a_θ(θ)) via chain rule                 │    │
│  │                                                                 │    │
│  │  === α Update ===                                               │    │
│  │    log α -= β_α · (E_tgt - sg(E_res))                          │    │
│  │                                                                 │    │
│  │  === Target Update ===                                          │    │
│  │    Q̄ ← (1-τ)Q̄ + τQ                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. RK-FAC vs NFT vs PPO: 关键差异

| 特性 | PPO | NFT | RK-FAC |
|------|-----|-----|--------|
| 信任域 | Clipping | L2 + max_drift | **E_res (精确 KL bound)** |
| Temporal Credit | GAE (需 V-network) | Terminal binary | **TD bootstrap (Q-network)** |
| Critic | V(s) 回传到 VLM | 无 | **Q(sg[h_VLM], a) frozen-VLM** |
| OOD 鲁棒性 | 差 (critic 过拟合视觉) | 好 (无 critic) | **好 (Q 不影响 VLM)** |
| 策略改进 | Logprob ratio | Contrastive energy | **Q-gradient through ODE** |
| 数据需求 | On-policy | On-policy (SDE chain) | **Semi-on-policy (可扩展off-policy)** |

## 4. 修改的文件一览

| 文件 | 修改内容 | ~Lines |
|------|---------|--------|
| `rlinf/algorithms/registry.py` | `"flow_rkfac"` 加入 bypass 列表 | +1 |
| `rlinf/models/embodiment/base_policy.py` | `ForwardType.RKFAC_ODE` | +1 |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | `forward_rkfac_ode()` + `_frozen_expert_velocities()` + dispatch | +130 |
| `rlinf/algorithms/losses.py` | `compute_residual_kinetic_energy()` + `flow_rkfac` loss placeholder | +45 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | RK-FAC 检测 + `collect_vlm_embedding` | +3 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | Init (Q, target Q, α, frozen ref) + credit assignment + training branch | +170 |
| `examples/embodiment/config/libero_object_rkfac_openpi_quickstart.yaml` | YAML 配置 | 新文件 |

**复用的现有组件**：
- `MultiQHead` (`q_head.py`) — 双Q网络，已有 state+action→Q 值
- `get_suffix_out()` — 单步 action expert forward（复用 VLM KV cache）
- `forward_gfn_chain()` — 1 VLM prefix + K suffix forwards 的模式参考
- `ForwardType.VELOCITY` — VLM prefix encoding + KV cache 生成

## 5. 关键实现细节

### 5.1 两阶段 ODE Forward (`forward_rkfac_ode`)

`forward_rkfac_ode()` 采用两阶段设计避免频繁权重交换：

```
Phase 1: ODE with current weights (K suffix forwards)
  for k in range(K):
    u_theta_k = model(x_k, t_k)          # 有梯度
    record (x_k, t_k) trajectory          # detach
    x_{k+1} = x_k + u_theta_k * dt       # Euler step

Phase 2: Frozen expert velocities (K suffix forwards, weight swap)
  save current expert weights
  load frozen expert weights
  for k in range(K):
    u_pre_k = model(x_k, t_k)            # 无梯度
  restore current expert weights

Phase 3: Compute E_res
  E_res = Σ_k ½ ||u_theta_k - u_pre_k||² · dt_k
```

**关键点**：权重交换只发生两次（load frozen + restore current），不是 2K 次。Phase 2 的所有 K 步共享同一次权重交换。

### 5.2 Frozen Expert Reference

初始化时保存 action expert + action_out_proj 的初始权重（不是 EMA，是固定不变的）：
```python
self._frozen_expert_state = {}
for name, param in model.named_parameters():
    if 'expert' in name or 'action_out' in name:
        self._frozen_expert_state[name] = param.data.detach().clone()
```

**与 NFT EMA 的区别**：NFT 用 EMA 追踪当前策略，RK-FAC 用固定的预训练权重。

### 5.3 Q-network 架构

复用 `MultiQHead`（双 Q 网络），输入：
- `vlm_embedding`: pooled VLM prefix output, [batch, 2048]，**detached**（Q 梯度不回传 VLM）
- `action_flat`: flattened action chunk, [batch, chunk × dim]

```
Q_j(sg[h_VLM(o)], a) → scalar
```

Q-network 参数量 ~2M（远小于 VLM 的 3B），训练开销可忽略。

### 5.4 Delayed Actor Update

```
for each mini-batch:
  1. Always: Critic update (Q loss)
  2. Every N steps: Actor update (ODE forward + E_res + Q gradient)
```

Critic 先学好 Q 值估计，再用 Q 梯度指导 actor → 训练更稳定。

## 6. 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `rkfac_alpha_init` | 0.1 | 初始温度 α（控制 E_res 惩罚强度）|
| `rkfac_alpha_lr` | 3e-4 | α 学习率 |
| `rkfac_e_tgt` | 1.0 | 目标残差动能 (C × d_a, C≈0.14, d_a=7) |
| `rkfac_gamma` | 0.99 | TD 折扣因子 |
| `rkfac_tau` | 0.005 | Target Q-network EMA rate |
| `rkfac_q_lr` | 1e-4 | Q-network 学习率 |
| `rkfac_q_hidden_dims` | [512, 256, 128] | Q MLP 隐藏层维度 |
| `rkfac_actor_delay` | 2 | Critic 更新 N 次后 actor 更新 1 次 |
| `rkfac_num_ode_steps` | 4 | Actor update 时的 ODE 步数 K |
| `rkfac_grad_clip` | 1.0 | Q-network 梯度裁剪 |
| `rkfac_vlm_hidden` | 2048 | VLM 嵌入维度 (PaliGemma 2B) |

**调参建议**：
- `rkfac_e_tgt`: 最关键参数。控制策略允许偏离预训练多少。太小 → 策略被锁定；太大 → 策略崩溃。从 1.0 开始，观察 `actor/e_res` 是否围绕 e_tgt 波动
- `rkfac_alpha_init`: 如果初始 E_res 远大于 e_tgt → 减小 alpha_init；反之增大
- `rkfac_q_lr`: Q 学习太快会导致过估计 → 减小；太慢导致 actor 信号弱 → 增大
- `rkfac_actor_delay`: 增大到 4 或 8 可以让 Q 更稳定后再更新 actor
- `rkfac_num_ode_steps`: 增大 K 提高 E_res 精度但增加计算量。4 是好的起点

## 7. 监控指标

TensorBoard 中关注：

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `actor/q_loss` | Q-network TD loss | 逐步下降（Q 学到 value）|
| `actor/actor_loss` | Actor loss (α·E_res - Q) | 下降（actor 改进）|
| `actor/e_res` | 残差动能 | 围绕 `e_tgt` 波动（α 自动调节）|
| `actor/alpha` | 温度参数 α | 先下降（允许偏离）后上升（收紧）|
| `actor/q_mean` | Q 值均值 | 从 0 增长（学到正值 Q for success）|
| `actor/td_target_mean` | TD target 均值 | 稳定增长 |
| `rkfac/success_rate` | 成功率 | 核心指标，稳步上升 |
| `rkfac/episode_reward_mean` | Episode 奖励均值 | 上升 |

**训练阶段**：
1. **前 50 步**: Q loss 下降、alpha 调整、E_res 稳定
2. **100 步后**: success_rate 开始上升
3. **稳定期**: E_res ≈ e_tgt, Q_mean 增长, success_rate 持续改善

**异常诊断**：
- E_res 持续增大 → alpha_init 太小或 e_tgt 太大
- Q loss 不降 → q_lr 太小或数据太少
- Q_mean 快速增大 → Q 过估计，增大 actor_delay 或减小 q_lr
- success_rate 不升 → 检查 Q_mean 是否有变化；如果 Q 在学但 actor 不动 → actor lr 太小

## 8. 一键启停

```bash
# 训练
bash run_embodiment.sh libero_object_rkfac_openpi_quickstart

# 评估
bash run_embodiment.sh libero_object_rkfac_openpi_quickstart --only_eval

# 恢复训练
bash run_embodiment.sh libero_object_rkfac_openpi_quickstart --resume_dir <path>
```

## 9. 关键公式速查

```
残差动能:     E_res = Σ_k ½ ||u_θ(x_k, t_k) - u_pre(x_k, t_k)||² · Δt_k
              x_{k+1} = x_k + u_θ(x_k, t_k) · Δt_k   (Euler ODE)
              Δt_k = t_k - t_{k+1},  schedule: linspace(1, 0, K+1)

KL bound:     D_KL(P_θ || P_pre) = E_res / σ²  (Girsanov, 精确)

Critic loss:  y = r + γ · min_j Q̄_j(s', a')
              L_Q = ½ Σ_j ||Q_j(s, a) - y||²

Actor loss:   a_θ = ODE(u_θ, ε; K)            [K步 Euler ODE]
              L_actor = α · E_res - min_j Q_j(vlm_emb, a_θ)

Alpha update: log α -= β_α · (E_tgt - sg(E_res))

Target update: Q̄ ← (1-τ)Q̄ + τQ

VLM embedding: vlm_emb = mean_pool(prefix_output).detach()
Q input:       Q(vlm_emb, flatten(a[:chunk, :dim]))
```

## 10. 理论基础

**Girsanov 定理**（详见 Research Proposal §3.2）：

对于 ODE 路径 dX = u_θ dt（从 N(0,I) 到动作空间），路径测度之间的 KL 散度：

```
D_KL(P_θ || P_pre) = (1/2σ²) ∫₀¹ E[||u_θ(X_t, t) - u_pre(X_t, t)||²] dt
```

当 σ → 0（ODE 极限），这个关系变为：
```
D_KL ≥ E_res / σ² → ∞
```

但 E_res 本身是一个**有限且有意义的度量**：它衡量速度场偏离程度，独立于 σ。

**直觉**：E_res 控制"策略偏离预训练多远"——不需要 log-likelihood 计算（flow matching 不提供），也不需要 clipping heuristic（PPO 需要）。
