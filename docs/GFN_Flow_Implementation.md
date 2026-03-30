# GFN-Flow (GFlowNet-Guided Denoising) — 实现文档

## 1. 算法概述

GFN-Flow 将 GFlowNet 的轨迹平衡框架应用于 flow-based VLA 的 RL 训练。与奖励最大化方法（PPO、FPI、AWM）不同，它学习 π(a|s) ∝ R(a,s) — **在提升性能的同时保持 action 多样性**。

K 步去噪过程被建模为 GFlowNet DAG，使用 Sub-Trajectory Balance (SubTB) 训练：

```
┌─────────────────────────────────────────────────────────┐
│ GFlowNet DAG (K=4 步去噪)                                │
│                                                          │
│  s_0 (noise)  →  s_1  →  s_2  →  s_3  →  s_4 (clean)   │
│  t=1.0          t=0.75   t=0.5    t=0.25   t=0.0        │
│      │P_F          │P_F      │P_F      │P_F              │
│      │             │         │         │                  │
│  F(s_0)         F(s_1)    F(s_2)    F(s_3)  F(s_4)=R     │
│      ← P_B ──── ← P_B ── ← P_B ── ← P_B                │
│                                                          │
│  SubTB: ∀(j,k): (log F_j + Σlog P_F - log F_k - Σlog P_B)² ≈ 0  │
│  Boundary: (log F(s_K) - log R)² ≈ 0                    │
└─────────────────────────────────────────────────────────┘
```

**核心特点**：
- **无 advantage 计算**（不同于 FPI/AWM）
- **无对比损失**（不同于 FlowSAR）
- **无 value network V(o,l)** — 替换为 State Flow Network F_ψ
- **SubTB loss** 替代 policy gradient / weighted MSE
- **保持多样性**：不会 mode collapse

## 2. 数据流

```
Rollout Worker                          Actor Worker
┌──────────────┐                        ┌────────────────────────────┐
│ π_θ rollout  │→ actions, rewards      │ Credit Assignment:         │
│              │                        │   log_R = log(max(R,ε))    │
│ EMA annotate:│                        │   (GFN requires R > 0)     │
│  1 VLM prefix│                        │                            │
│  K suffix    │→ gfn_chain [K+1]       │ Training (per mini-batch): │
│  1 suffix    │→ v_old (trust region)  │  ForwardType.GFN_CHAIN:    │
│              │→ flow_t, flow_epsilon  │   1 VLM + (K+2) suffix    │
└──────────────┘                        │   → velocities [K]         │
                                        │   → log_flows [K+1]        │
                                        │   → v_reg (trust region)   │
                                        │                            │
                                        │  log P_F ← velocities     │
                                        │  log P_B ← fixed backward │
                                        │  SubTB + boundary loss     │
                                        │  + KL trust region         │
                                        └────────────────────────────┘
```

## 3. GFN-Flow vs 其他方法

| 特性 | FlowIPO | FlowSAR | FPI | AWM-VLA | **GFN-Flow** |
|------|---------|---------|-----|---------|-------------|
| 核心思想 | ODE 插值 + 对比 | 自标注 + 能量加权 | exp(A/λ) 加权 | 归一化 A 加权 | **SubTB 流平衡** |
| Value Network | 否 | 否 | 是 | 是 | **否 (用 F_ψ)** |
| 信用分配 | δ_i 插值权重 | 重建误差 e_i | exp(A_i/λ) | clip(normalize(A_i)) | **隐式 (SubTB)** |
| 多样性保持 | 否 | 否 | 否 | 否 | **是** |
| Loss 形式 | Interpolated FM | Energy-weighted FM | w · MSE | Ã · MSE | **SubTB 残差²** |
| 理论保证 | 启发式 | 启发式 | Policy improvement | ≈ Policy gradient | **π ∝ R (精确)** |

## 4. 时间约定

代码使用 t (t=0→clean, t=1→noise)，与所有其他 Flow 方法一致：

| 概念 | GFN 论文 (τ) | 代码 (t) |
|------|-------------|---------|
| Clean data (action) | τ = 1 | t = 0 |
| Pure noise | τ = 0 | t = 1 |
| 插值 | x_τ = τ·a + (1-τ)·ε | x_t = (1-t)·a + t·ε |
| 目标 velocity | a - ε | ε - a |
| Euler step | x_{k+1} = x_k + v·δ | x_{k+1} = x_k - v·δ |

## 5. State Flow Network F_ψ

```
suffix_out (action expert) → [detach] → mean pool → + sinusoidal(t) → MLP → scalar
                                                                        ↓
                                                              [1088 → 512 → 256 → 128 → 1]
                                                                        ↓
                                                              log F_ψ(x_k, t_k, s)
```

关键设计：
- 输入 suffix_out 包含 observation（via VLM KV cache）和 action state（x_k, t_k）信息
- `detach_critic_input: True`：F_ψ 梯度不穿透 action expert backbone
- 参数量 ~700K（相比 π0 action expert 300M 可忽略）
- 使用 GELU 激活，Kaiming normal 初始化，输出层小 std (0.02)

## 6. SubTB Loss

对于 K+1 个状态 (s_0, ..., s_K)，枚举所有 C(K+1, 2) 个子轨迹对 (j, k)：

```
residual(j,k) = log F(s_j) + Σ_{i=j}^{k-1} log P_F(i)
              - log F(s_k) - Σ_{i=j}^{k-1} log P_B(i)

L_SubTB = Σ_{j<k} λ^{k-j} · residual² / Σ λ^{k-j}

L_boundary = (log F(s_K) - log R)²

L_total = L_SubTB + α · L_boundary + β · ||v_θ - v_old||²
```

对于 K=4：C(5,2) = 10 个子轨迹对。使用累积和技巧高效计算。

## 7. Forward/Backward Policy

### Forward P_F (可学习, 通过 v_θ)

```
P_F(x_{k+1} | x_k) = N(μ_k, σ_f²I)
μ_k = x_k - v_θ(x_k, t_k, s) · δ_k   (Euler step)
δ_k = t_k - t_{k+1} > 0               (timestep 递减)
```

### Backward P_B (固定, 无可学习参数)

```
P_B(x_k | x_{k+1}) = N(μ_B, σ_b²I)
μ_B = ((1-t_k) / (1-t_{k+1})) · x_{k+1}   (rectified flow 反向缩放)
```

- ratio = (1-t_k)/(1-t_{k+1}) < 1（因为 t_k > t_{k+1}）
- 特殊情况：t_k=1 → ratio=0 → μ_B=0（反向到纯噪声）

## 8. 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `rlinf/models/embodiment/modules/state_flow_net.py` | **新建** StateFlowNet F_ψ MLP |
| `rlinf/models/embodiment/base_policy.py` | `ForwardType.GFN_CHAIN` 枚举值 |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | Config + init + `forward_gfn_chain()` |
| `rlinf/models/embodiment/openpi/__init__.py` | StateFlowNet 转 bfloat16（FSDP dtype 一致性） |
| `rlinf/algorithms/losses.py` | `compute_flow_gfn_loss()` — SubTB + boundary |
| `rlinf/algorithms/credit_assignment.py` | `compute_gfn_log_pf()` + `compute_gfn_log_pb()` |
| `rlinf/algorithms/registry.py` | `"flow_gfn"` 加入 bypass 列表 |
| `rlinf/hybrid_engines/fsdp/utils.py` | StateFlowNet FSDP wrap policy（与 ValueHead 同模式） |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | GFN 检测 + EMA + `_compute_gfn_annotations()` |
| `rlinf/workers/actor/fsdp_actor_worker.py` | GFN 初始化 + 信用分配 + 训练分支 |
| `examples/embodiment/config/model/pi0.yaml` | 添加 `add_state_flow_net` 字段（默认 False） |
| `examples/embodiment/config/libero_object_flowgfn_openpi_quickstart.yaml` | **新建** YAML 配置 |

## 9. 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `flow_gfn_denoise_steps` | 4 | K: DAG 深度（去噪步数） |
| `flow_gfn_sigma_f` | 0.1 | Forward policy 噪声 σ_f |
| `flow_gfn_sigma_b` | 0.1 | Backward policy 噪声 σ_b |
| `flow_gfn_subtb_lambda` | 1.0 | SubTB 几何权重 λ^length |
| `flow_gfn_boundary_coeff` | 1.0 | Boundary loss 系数 α |
| `flow_gfn_kl_coeff` | 0.5 | Trust region KL 系数 β |
| `flow_gfn_flow_loss_coeff` | 1.0 | SubTB + boundary loss 总系数 |
| `flow_gfn_ema_beta` | 0.995 | EMA 衰减率 |
| `flow_gfn_warmup` | 5 | Warmup epochs（纯 BC + F_ψ 训练） |
| `flow_gfn_reward_floor` | 0.01 | Floor reward ε（GFN 要求 R > 0） |
| `detach_critic_input` | True | 阻止 F_ψ 梯度穿透 action expert |

**调参建议**：
- `sigma_f/sigma_b`：从 0.1 开始。过大→SubTB 残差噪声大；过小→探索不足
- `subtb_lambda`：1.0 = 所有子轨迹等权。< 1.0 更关注短子轨迹
- `boundary_coeff`：保持 1.0。如果 log F(s_K) 不收敛到 log R，增大
- `kl_coeff`：0.5 是保守值。如果策略不更新，减小；如果漂移太快，增大
- `warmup`：5 是最低限。F_ψ 需要时间学到有意义的 flow 估计
- `reward_floor`：0.01 保证 log R 不为 -∞。太大→失败轨迹也被学习

## 10. 一键启停

```bash
# 训练
bash run_embodiment.sh libero_object_flowgfn_openpi_quickstart

# 评估
bash run_embodiment.sh libero_object_flowgfn_openpi_quickstart --only_eval

# 恢复训练
bash run_embodiment.sh libero_object_flowgfn_openpi_quickstart --resume_dir <path>
```

## 11. 监控指标

TensorBoard 中关注：

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `gfn_loss` | SubTB + boundary 总 loss | 下降 |
| `gfn_subtb_loss` | SubTB 残差 loss | 下降 → 0 |
| `gfn_boundary_loss` | (log F(s_K) - log R)² | 下降 → 0 |
| `gfn_kl_loss` | Trust region ‖v_θ - v_old‖² | 小且稳定 |
| `gfn_log_flow_mean` | 平均 log F_ψ | 收敛 |
| `gfn_log_flow_terminal` | 终端 log F_ψ | → log R |
| `gfn_log_pf_mean` | 平均 log P_F | 稳定 |
| `gfn_log_pb_mean` | 平均 log P_B | 稳定 |
| `gfn/episode_reward` | Episode 平均奖励 | 上升 |
| `gfn/success_rate` | 成功率 | 上升 |
| `gfn/warmup_remaining` | 剩余 warmup epochs | 递减到 0 |

**训练阶段**：
1. **Warmup (前 N epochs)**：纯 BC，F_ψ 学习基本 flow 结构。KL loss 为 0。
2. **过渡期 (N ~ N+5)**：SubTB 开始生效，flow 值开始分化。
3. **稳定改进期**：SubTB 残差 → 0，boundary loss → 0，log F(s_K) ≈ log R，success_rate 上升。

**异常诊断**：
- `subtb_loss` 不下降 → σ_f/σ_b 太大，减小到 0.05
- `boundary_loss` 不收敛 → F_ψ 学习率太低或太高
- `kl_loss` 持续增大 → 策略漂移，增大 `kl_coeff`
- Success rate 震荡 → 增大 `kl_coeff` 或减小 `flow_loss_coeff`
- Success rate 不变 → warmup 太长或 SubTB loss 权重太小

## 12. 关键公式速查

```
DAG 结构:     s_0(noise) → s_1 → ... → s_K(clean)
Timesteps:    t_0=1 > t_1 > ... > t_K=0

Forward:      P_F(s_{k+1}|s_k) = N(x_k - v_θ·δ_k, σ_f²I)
Backward:     P_B(s_k|s_{k+1}) = N((1-t_k)/(1-t_{k+1})·x_{k+1}, σ_b²I)
Flow:         F_ψ(x_k, t_k, s) = StateFlowNet(pooled_suffix, t_k)

SubTB:        ∀(j<k): residual = log F_j + Σlog P_F - log F_k - Σlog P_B
              L_SubTB = Σ λ^{k-j} · residual² / Σ λ^{k-j}

Boundary:     L_boundary = (log F(s_K) - log R)²
Trust Region: L_KL = ||v_θ - v_old||²

Total:        L = L_SubTB + α·L_boundary + β·L_KL
```

## 13. 梯度分析

```
∇_θ L_SubTB ← via log P_F (uses v_θ for Euler step μ_k)
∇_ψ L_SubTB ← via log F_ψ (State Flow Network)
∇_θ L_KL    ← via v_θ at (x_t_reg, t_reg)
```

- θ (policy) 通过 log P_F 中的 Euler step 获得梯度
- ψ (flow) 通过 log F_ψ 获得梯度
- 分离的、干净的梯度路径（detach_critic_input 阻止 F_ψ → action expert）

收敛时：F_ψ(s) ∝ R(s) → π(a|s) ∝ R(a,s)，实现 reward-proportional sampling。
