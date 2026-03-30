# Hinge-NFT — 实现文档

## 1. 算法概述

Hinge-NFT = π-StepNFT的mirror construction + sample-space energy + **hinge margin loss** + **VLM embedding步级信用分配**。

```
L_total = max(0, m + ½ y ΔE)  +  β_kl · ||v_θ - v_old||²
          ↑ hinge contrastive    ↑ trust region (velocity KL)
```

与π-StepNFT的**两个差别**：
1. softplus → hinge（对称性，允许连续advantage）
2. terminal binary → VLM embedding change rate 步级信用分配

**两种运行模式**：

| 模式 | `adv_type` | 信用分配 | 额外成本 | 描述 |
|------|-----------|---------|---------|------|
| embedding_change（默认） | `embedding_change` | 连续 $y \in [-1, 1]$ | **零**（embedding已在rollout时收集） | 完整方案：hinge + 步级信用 |
| terminal_binary | `terminal_binary` | $y = \pm 1$ | 零 | 消融实验：只测hinge loss效果 |

## 2. 数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Rollout Worker                                │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ π_θ rollout — 独立 NFT SDE 去噪分支（复用 flow_nft）      │           │
│  │  ⚠️ 所有 K 步均使用 mode="train" (SDE 噪声)               │           │
│  └──────────────────────────────────────────────────────────┘           │
│        │                                                                 │
│        ▼  随机选一步 k ∈ [0, K)                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ sample_actions 返回 (与 flow_nft 完全相同 + vlm_embedding):  │       │
│  │  nft_xt, nft_v, nft_xnext, nft_step_index, nft_noise_level │       │
│  │  vlm_embedding = mean_pool(prefix_output).detach()          │       │
│  │                                                              │       │
│  │  ★ 零额外计算：embedding 在 VLM prefix 编码后顺手收集       │       │
│  │  ★ 无需 EMA 参考模型                                         │       │
│  │  ★ 无需额外 forward pass                                     │       │
│  └──────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  send to actor
┌──────────────────────────────────────────────────────────────────────────┐
│                           Actor Worker                                   │
│                                                                          │
│  ┌─── Credit Assignment (方案B: VLM Embedding Change Rate) ──────┐    │
│  │  embedding_change (默认):                                       │    │
│  │    w_t = ||e_{t+1} - e_t||₂  (帧间 L2 距离)                    │    │
│  │    w_norm = clamp(w / mean(w), w_min, w_max)                    │    │
│  │    A_t = sign(R) × w_norm × c_adv                              │    │
│  │                                                                  │    │
│  │  直觉:                                                          │    │
│  │    手臂接近物体 → ||Δe|| 小 → 低权重 (routine)                  │    │
│  │    抓取/放置    → ||Δe|| 大 → 高权重 (关键决策点)               │    │
│  │                                                                  │    │
│  │  terminal_binary (消融):                                        │    │
│  │    y = +1 (success), y = -1 (failure), broadcast all steps      │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│        │                                                                │
│        ▼                                                                │
│  ┌─── Training (per mini-batch, SINGLE VLM forward) ─────────────┐    │
│  │  ForwardType.VELOCITY(x_t=nft_xt, timestep=t_k):               │    │
│  │    VLM prefix encoding → [一次 VLM forward]                     │    │
│  │    suffix → velocity head → v_θ   (有梯度)                     │    │
│  │                                                                 │    │
│  │  Mirror Construction (与 NFT 相同):                             │    │
│  │    Δv_clip = clip(v_θ - v_old, max_drift)                      │    │
│  │    v⁺ = v_old + β · Δv_clip                                    │    │
│  │    v⁻ = v_old - β · Δv_clip                                    │    │
│  │                                                                 │    │
│  │  Sample-space Mahalanobis Energy (与 NFT 相同):                 │    │
│  │    E⁺, E⁻, ΔE = E⁺ - E⁻                                      │    │
│  │                                                                 │    │
│  │  ★ Hinge Loss (核心改动):                                      │    │
│  │    logit = ½ · y · ΔE                                           │    │
│  │    L_hinge = max(0, margin + logit)                             │    │
│  │    L_KL  = ||v_θ - v_old||²                                    │    │
│  │    L = L_hinge + kl_beta · L_KL                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. 修改的文件一览

| 文件 | 修改内容 |
|------|---------|
| `rlinf/algorithms/losses.py` | 新增 `compute_flow_hinge_nft_loss()` |
| `rlinf/algorithms/credit_assignment.py` | 新增 `compute_embedding_change_advantages()` |
| `rlinf/algorithms/registry.py` | `"flow_hinge_nft"` 加入 bypass 列表 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | Hinge-NFT 初始化 + 信用分配 + 训练分支 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | Hinge-NFT 检测（无需 EMA/annotation） |
| `examples/embodiment/config/libero_object_hinge_nft_openpi_quickstart.yaml` | YAML 配置 |

**无需修改**：
- `openpi_action_model.py` — 复用 `use_nft_loss=True` 的 SDE 快照收集 + vlm_embedding 输出

## 4. 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `flow_hinge_nft_margin` | 1.0 | Hinge margin $m$ |
| `flow_hinge_nft_beta` | 1.0 | 镜像构造缩放 β |
| `flow_hinge_nft_kl_beta` | 0.0001 | KL trust region 权重 |
| `flow_hinge_nft_max_drift` | 0.5 | 速度变化裁剪 |
| `flow_hinge_nft_noise_level` | 0.2 | SDE 探索噪声 |
| `flow_hinge_nft_adv_clip_max` | 1.0 | Advantage 裁剪 |
| `flow_hinge_nft_adv_type` | `embedding_change` | `embedding_change` 或 `terminal_binary` |
| `flow_hinge_nft_emb_w_min` | 0.2 | Embedding credit 权重下界 |
| `flow_hinge_nft_emb_w_max` | 2.0 | Embedding credit 权重上界 |

**调参建议**：
- `margin`：核心新参数。1.0 是起点。如果 hinge_loss 不下降 → 减小到 0.5（更容易满足margin）；如果 margin_violation 太高 → 增到 2.0
- `max_drift`：最关键的稳定性参数，与 NFT 一样。0.5 是起点
- `emb_w_min`：保证 routine step 至少有 20% 信号强度（$|y| \geq 0.2$）
- `emb_w_max`：限制关键 step 最多 200% 信号强度，防止单步主导

## 5. 监控指标

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `hinge_loss` | Hinge 对比 loss | 下降 |
| `kl_loss` | Trust region | 小且稳定 |
| `total_loss` | hinge_loss + kl_beta * kl_loss | 下降 |
| `margin_violation_frac` | margin 未满足的比例 | 从 ~1.0 下降（模型逐渐学会） |
| `margin_satisfied_frac` | margin 已满足（loss=0）的比例 | 上升 |
| `active_loss_mean` | margin 未满足样本的平均 loss | 下降 |
| `pref_acc` | $y \cdot \Delta E < 0$ 的比例（正确分类） | 上升 |
| `delta_E_mean` | $E^+ - E^-$ 均值 | 趋向负值 |
| `delta_v_norm_mean` | $\|v_\theta - v_{\text{old}}\|$ | 适度增长 |
| `clip_frac` | max_drift 裁剪比例 | 适中 (0.2-0.5) |
| `adv_mean` / `adv_std` | Advantage 统计 | 随成功率变化 |
| `y_abs_mean` | $|y|$ 均值（embedding_change 下 ≈ 0.2~2.0） | 有方差（步间有区分度） |
| `env/success_once` | 首次成功率 | 上升 |

**训练阶段**：
1. **初始期**：margin_violation ≈ 1.0（所有样本违反margin）。pref_acc ≈ 0.5。正常。
2. **信号涌现**：pref_acc > 0.5，margin_satisfied 开始上升，success_once 开始上升。
3. **稳定期**：margin_satisfied > 0.3，pref_acc > 0.7。

## 6. 关键公式速查

```
Mirror:           Δv = v_θ - v_old
                  Δv_clip = Δv · min(max_drift / ||Δv||, 1)
                  v⁺ = v_old + β · Δv_clip
                  v⁻ = v_old - β · Δv_clip

Flow Mean:        μ = flow_mean(x_t, v)    (与 NFT 完全相同)

Energy:           σ² = (√δ · σ_i)² + ε
                  E = Σ_dim (x_next - μ)² / σ²

★ Hinge Loss:    y = clip(advantage / c, -1, 1)
                  logit = ½ · y · (E⁺ - E⁻)
                  L_hinge = max(0, margin + logit)
                  L_KL = ||v_θ - v_old||²
                  L = L_hinge + kl_beta · L_KL

★ Embedding       w_t = ||e_{t+1} - e_t||₂
  Change Credit:   w_norm = clamp(w / mean(w), w_min, w_max)
                   A_t = sign(R) × w_norm × c_adv
```

## 7. 一键启停

```bash
# 训练（embedding_change 模式 — 默认，零额外成本）
bash run_embodiment.sh libero_object_hinge_nft_openpi_quickstart

# 训练（terminal_binary 模式 — 消融实验）
bash run_embodiment.sh libero_object_hinge_nft_openpi_quickstart \
  algorithm.flow_hinge_nft_adv_type=terminal_binary
```
