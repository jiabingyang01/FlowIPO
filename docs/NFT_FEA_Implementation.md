# π-StepNFT — 实现文档

## 1. 算法概述

π-StepNFT 的核心思想：**用对比镜像构造 + sample-space Mahalanobis 能量做 critic-free 的 flow RL**。

```
L_total = softplus(½ y (E⁺ - E⁻))  +  β_kl · ||v_θ - v_old||²
          ↑ NFT contrastive loss        ↑ trust region (velocity KL)
```

关键创新（对比 FlowSAR/AWM）：

- **梯度不依赖 ||v_θ - u||²**：梯度 ∝ (v_θ - v_old)，从零开始增长，不随预训练衰减
- **无需 Value Network**：使用 terminal binary advantage（成功 +1，失败 -1）
- **Sample-space 能量 + Mahalanobis 归一化**：各噪声层贡献均等
- **实际 SDE 链快照**：训练数据来自 rollout 的真实去噪链，不是随机 (t, ε)

> **注意**：FEA（Frozen Embedding Advantage）代码保留在codebase中但**不推荐使用**。实验证明FEA与softplus contrastive loss不兼容——详见Idea文档§3.3和§5.2.1。

## 2. 数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Rollout Worker                                │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ π_θ rollout — 独立 NFT SDE 去噪分支                       │           │
│  │                                                            │           │
│  │  ⚠️ 关键实现细节：                                          │           │
│  │  NFT 使用独立的 SDE 循环 (collect_flow_snap=True)          │           │
│  │  所有 K 步均使用 mode="train" (SDE 噪声)                   │           │
│  │  不与正常 ODE 推理循环共享！                                 │           │
│  │  完成后直接 return，不经过 chains/denoise_inds 逻辑          │           │
│  └──────────────────────────────────────────────────────────┘           │
│        │                                                                 │
│        ▼  随机选一步 k ∈ [0, K)                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ NFT 快照数据 (sample_actions 返回):                          │       │
│  │  nft_xt      = x at step k        (denoising 状态)          │       │
│  │  nft_v       = v_θ at step k      (v_old, detach)           │       │
│  │  nft_xnext   = x after step k     (含 SDE 噪声)            │       │
│  │  nft_step_index = k               (用于 schedule 查询)      │       │
│  │  nft_noise_level                   (SDE 噪声水平)           │       │
│  │  vlm_embedding = pool(prefix_output).detach()  (备用)       │       │
│  │                                                              │       │
│  │  NO annotation pass (不需要 EMA 模型！)                      │       │
│  │  NO chains / denoise_inds (NFT 不需要)                       │       │
│  └──────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  send to actor
┌──────────────────────────────────────────────────────────────────────────┐
│                           Actor Worker                                   │
│                                                                          │
│  ┌─── Credit Assignment ───────────────────────────────────────────┐    │
│  │  terminal_binary (推荐 ✅):                                      │    │
│  │    y = +1 (success), y = -1 (failure)                           │    │
│  │    broadcast to all env steps                                    │    │
│  │                                                                  │    │
│  │  frozen_embedding (不推荐 ⚠️):                                   │    │
│  │    Ridge regression → V_probe(e_t) → |TD error| 重要性          │    │
│  │    方向来自 terminal_binary，FEA 只调节幅度                      │    │
│  │    （softplus contrastive 与连续 advantage 不兼容，详见 Idea 文档）│    │
│  └──────────────────────────────────────────────────────────────────┘    │
│        │                                                                │
│        ▼                                                                │
│  ┌─── Training (per mini-batch, SINGLE VLM forward) ─────────────┐    │
│  │                                                                 │    │
│  │  ForwardType.VELOCITY(x_t=nft_xt, timestep=t_k):               │    │
│  │    VLM prefix encoding → [一次 VLM forward]                     │    │
│  │    suffix → velocity head → v_θ   (有梯度, 训 policy)          │    │
│  │                                                                 │    │
│  │  Mirror Construction:                                           │    │
│  │    Δv = v_θ - v_old                                             │    │
│  │    Δv_clip = Δv · min(max_drift / ||Δv||, 1)                   │    │
│  │    v⁺ = v_old + β · Δv_clip                                    │    │
│  │    v⁻ = v_old - β · Δv_clip                                    │    │
│  │                                                                 │    │
│  │  Sample-space Mahalanobis Energy:                               │    │
│  │    μ⁺ = flow_mean(x_t, v⁺),  μ⁻ = flow_mean(x_t, v⁻)        │    │
│  │    σ² = sqrt(δ) · σ_i · noise_level                            │    │
│  │    E⁺ = Σ (x_next - μ⁺)² / σ²                                 │    │
│  │    E⁻ = Σ (x_next - μ⁻)² / σ²                                 │    │
│  │                                                                 │    │
│  │  NFT Loss:                                                      │    │
│  │    L_NFT = softplus(½ · y · (E⁺ - E⁻))                        │    │
│  │    L_KL  = ||v_θ - v_old||²                                    │    │
│  │    L = L_NFT + kl_beta · L_KL                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. NFT vs FlowSAR：关键差异

| 特性 | FlowSAR (softplus_kl) | NFT-FEA |
|------|----------------------|---------|
| 镜像构造 | v⁺ = (1-β)v_old + βv_θ | v⁺ = v_old + β·Δv_clip |
| 能量空间 | Velocity space: \|\|v - u\|\|² | **Sample space: \|\|x_next - μ\|\|²** |
| 方差归一化 | 无 或 1/(2t) | **Mahalanobis σ² = δ·σ_i²** |
| 数据来源 | 随机 (t, ε), 需要 EMA 标注 | **实际 SDE 链快照，无需 EMA** |
| 最大漂移控制 | 无 | **max_drift 裁剪 Δv** |
| Value Network | 不需要 | **不需要** |
| 梯度幅度 | O(ε) (随预训练衰减) | **O(1) (始终存在)** |
| Annotation Pass | 需要 (EMA 模型 2 次前向) | **不需要** |

## 4. 时间约定

与 FlowIPO 一致，代码使用 t (t=0→clean, t=1→noise)：

| 概念 | 代码 (t) | 说明 |
|------|---------|------|
| Clean data | t = 0 | x₀ = action |
| Pure noise | t = 1 | x₁ = ε ~ N(0,I) |
| Schedule | linspace(1, 0, K+1) | 从噪声到干净 |
| SDE 步进 | x_{k+1} = flow_mean + σ·z | Euler-Maruyama |

## 5. 关键实现细节

### 5.1 独立 SDE 去噪分支（关键修复）

`sample_actions()` 中 NFT 使用**独立的去噪循环**，不与正常 ODE 推理共享。

**原因**：π-StepNFT 要求所有 K 步去噪都使用 SDE 噪声（`mode="train"`），但正常推理循环根据 `denoise_steps` 配置只在部分步骤加噪声。如果共享循环，NFT 快照数据的分布与训练时 loss 计算的假设不匹配，导致训练崩溃。

```python
# openpi_action_model.py, sample_actions() 内部
if collect_flow_snap:
    # 独立循环：所有步骤 mode="train"
    for idx in range(num_steps):
        x_t_mean, x_t_std, value_t, v_t = self.sample_mean_var_val(
            x_t, idx, state, prefix_pad_masks, past_key_values,
            mode="train",  # ← 关键：ALL steps use SDE noise
            denoise_steps=num_steps,
            compute_values=False,
        )
        # 保存随机选中步骤的快照
        mask = flow_rand_idx == idx
        if mask.any():
            flow_xt_snap[mask] = x_t.detach()[mask]
            flow_v_snap[mask] = v_t.detach()[mask]
        # Euler + SDE noise step
        x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
        if mask.any():
            flow_xnext_snap[mask] = x_t.detach()[mask]
    # 直接 return，不经过 chains/denoise_inds 逻辑
    return {"actions": x_0, "nft_xt": ..., "nft_v": ..., ...}
```

### 5.2 predict_action_batch() 条件访问

由于 NFT 分支提前返回不包含 `chains`/`denoise_inds`，`predict_action_batch()` 需要条件访问：

```python
if "chains" in outputs:
    forward_inputs["chains"] = outputs["chains"]
    forward_inputs["initial_noise"] = outputs["chains"][:, 0]
if "denoise_inds" in outputs:
    forward_inputs["denoise_inds"] = outputs["denoise_inds"]
```

## 6. 修改的文件一览

| 文件 | 修改内容 |
|------|---------|
| `rlinf/algorithms/losses.py` | `compute_flow_nft_loss()` — 对比镜像 + sample-space Mahalanobis |
| `rlinf/algorithms/credit_assignment.py` | `compute_terminal_binary_advantages()` + `compute_frozen_embedding_advantages()` |
| `rlinf/algorithms/registry.py` | `"flow_nft"` 加入 bypass 列表 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | NFT 初始化 + 信用分配 + 训练分支 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | NFT 检测 (无需 EMA/annotation) |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | NFT 快照收集 + VLM 嵌入池化 |
| `examples/embodiment/config/libero_object_flownft_openpi_quickstart.yaml` | YAML 配置 |

**无需修改的基础设施**：
- `base_policy.py` — `ForwardType.VELOCITY` 枚举值（复用）
- `openpi_action_model.py` — `forward_velocity()` 已有，直接使用

## 7. 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `flow_nft_beta` | 1.0 | 镜像构造缩放 β (与 π-StepNFT 论文一致) |
| `flow_nft_kl_beta` | 0.0001 | KL trust region 权重 (很小，主靠 max_drift) |
| `flow_nft_max_drift` | 0.5 | 速度变化裁剪 (关键稳定性参数) |
| `flow_nft_dpo_beta` | 1.0 | softplus logit 缩放（原论文固定为1） |
| `flow_nft_noise_level` | 0.2 | SDE 探索噪声 |
| `flow_nft_adv_clip_max` | 1.0 | Advantage 裁剪 |
| `flow_nft_adv_type` | terminal_binary | 信用分配类型 (terminal_binary 或 frozen_embedding) |
| `flow_nft_fea_gamma` | 0.99 | FEA 时间折扣因子 |
| `flow_nft_fea_ridge_lambda` | 1.0 | Ridge 回归正则化 |

**调参建议**：
- `max_drift`：最关键参数。0.5 是起点。如果训练不稳定（logit 发散）→ 减小到 0.3；如果太保守 → 增到 0.8
- `kl_beta`：0.0001 很小是因为 max_drift 已提供主要约束。如果 kl_loss 飙升 → 增到 0.001
- `dpo_beta`：控制 softplus logit 缩放。1.0 是原论文默认；如果 pref_acc 快速饱和到 1 → 减到 0.5
- `noise_level`：SDE 探索噪声。太低→无探索；太高→行为策略太随机。0.2 是保守起点
- `adv_type`：**推荐 terminal_binary**。frozen_embedding (FEA) 代码保留但不推荐——softplus contrastive loss 对标签噪声极度敏感，FEA 产生的连续 advantage 会导致训练崩溃（详见 Idea 文档 §3.3）

## 8. 监控指标

TensorBoard 中关注：

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `nft_loss` | NFT softplus 对比 loss | 下降 (从 ~0.69 开始) |
| `kl_loss` | Trust region \|\|v_θ - v_old\|\|² | 小且稳定 |
| `total_loss` | nft_loss + kl_beta * kl_loss | 下降 |
| `pref_acc` | 对比偏好准确率 | 从 ~0.5 上升到 >0.7 |
| `pref_acc_strong` | 强信号偏好准确率 (\|y\|>0.3) | > pref_acc |
| `logit_mean` | 对比 logit 均值 | 负值且幅度增大 |
| `delta_E_mean` | E⁺ - E⁻ 差值 | 负值 (好的比差的更靠近) |
| `delta_v_norm_mean` | \|\|v_θ - v_old\|\| 平均范数 | 适度增长 |
| `clip_frac` | max_drift 裁剪比例 | 适中 (0.2-0.5) |
| `clip_coef_mean` | 平均裁剪系数 | 适中 |
| `adv_mean` | Advantage 均值 | 随成功率变化 |
| `E_pos_mean` / `E_neg_mean` | 正负分支能量 | E_pos < E_neg (好的更近) |
| `std_mean` / `std_min` / `std_max` | Mahalanobis 标准差 | 稳定 |
| `z2_mean` | 归一化残差的 L2 均值 | ~1 (Mahalanobis 工作) |
| `finite_frac` | ΔE 有限值比例 | ~1.0 (数值稳定) |
| `env/success_once` | 首次成功率 | 上升 |
| `env/episode_reward` | Episode 奖励 | 上升 |

**训练阶段**：
1. **初始期 (前 ~50 步)**：pref_acc ≈ 0.5（随机）。nft_loss ≈ 0.69 (= log 2)。这是正常的——模型还没学到区分好坏。
2. **信号涌现 (50-150 步)**：pref_acc 开始上升。delta_E_mean 变负。success_once 开始上升。
3. **稳定改进期**：pref_acc > 0.7，kl_loss 稳定，success_rate 稳步提升。

**异常诊断**：
- `pref_acc` 不上升 → 检查 advantage 信号（episode_reward 是否有方差？全成功或全失败都无信号）
- `kl_loss` 持续增大 → 增大 `kl_beta` 或减小 `max_drift`
- `clip_frac` ≈ 1.0 → `max_drift` 太小，增大到 0.8
- `clip_frac` ≈ 0.0 → 模型变化极小，可能 lr 太低
- `finite_frac` < 1.0 → 数值问题，检查 `std_epsilon` 和 `noise_level`
- `nft_loss` 不下降但 `kl_loss` 增大 → 策略在漂移但没学到有用信号

## 9. 一键启停

```bash
# 训练
bash run_embodiment.sh libero_object_flownft_openpi_quickstart

# 评估
bash run_embodiment.sh libero_object_flownft_openpi_quickstart --only_eval

# 恢复训练
bash run_embodiment.sh libero_object_flownft_openpi_quickstart --resume_dir <path>
```

## 10. 关键公式速查

```
Mirror:           Δv = v_θ - v_old
                  Δv_clip = Δv · min(max_drift / ||Δv||, 1)
                  v⁺ = v_old + β · Δv_clip
                  v⁻ = v_old - β · Δv_clip

Flow Mean:        x0_pred = x_t - v · t
                  x1_pred = x_t + v · (1 - t)
                  x0_weight = 1 - (t - δ)
                  x1_weight = t - δ - σ_i² · δ / (2t)
                  μ = x0_pred · x0_weight + x1_pred · x1_weight

Energy:           σ² = (√δ · σ_i)² + ε
                  E = Σ_dim (x_next - μ)² / σ²

NFT Loss:         y = clip(advantage / c, -1, 1)
                  logit = ½ · y · (E⁺ - E⁻)
                  L_NFT = softplus(logit)
                  L_KL = ||v_θ - v_old||²
                  L = L_NFT + kl_beta · L_KL

Terminal Binary (推荐):
                  y = +1 (success), -1 (failure)
                  broadcast to all env steps

FEA (不推荐，代码保留):
                  G_t = γ^(T-t) · R                (MC return)
                  w = (E^T E + λI)^{-1} E^T G      (Ridge)
                  V(e_t) = w^T e_t + b
                  td_t = γ V(e_{t+1}) - V(e_t)     (TD error)
                  importance = |td_t| / mean(|td|)  (归一化重要性)
                  importance = clamp(importance, 0.2, 2.0)
                  A_t = terminal_sign × importance × adv_clip_max
                  (方向始终来自 terminal binary, FEA 仅调节幅度)
```

## 11. NFT 梯度分析

NFT 的梯度（Theorem 4.4，π-StepNFT 论文）：

```
∇_θ L_NFT ∝ σ(z_t) · y · (∂v_θ/∂θ)^T · B_t · Σ_t^{-1} · e_t
```

其中 e_t 是 SDE 噪声残差，**始终为 O(σ)**，不随模型收敛而消失。

对比加权回归 (AWM/FPI)：
```
∇_θ L_AWM = E[Ã · 2(v_θ - u) · ∇_θ v_θ]
                          ↑ 这项对 on-policy data → 0
```

这是 NFT 收敛而 AWM/FPI 不收敛的数学根因。
