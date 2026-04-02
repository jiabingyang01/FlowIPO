# DECO — 实现文��

## 1. 算法概述

DECO (Deviation-Enhanced Contrastive Optimization) 是 π-StepNFT 的**严格超集**。唯一改动：将 episode 级标量标签 $y = 2r-1$ 替换为步级向量标签 $y_i = (2r-1)(1 + \eta \cdot w_i)$，其中 $w_i$ 来自冻结参考模型的速度偏差。

```
L_total = softplus(½ y_i (E⁺ - E⁻))  +  β_kl · ||v_θ - v_old||²
          ↑ DECO contrastive loss         ↑ trust region (velocity KL)
          │                                │
          └─ 唯一改动: y → y_i             └─ 不变
```

核心创新（对比 π-StepNFT）：

- **步级信用分配**：高偏差步（关键决策点）获得更大梯度权重
- **无 Critic**：偏差信号来自冻结参考模型的单次前向，不引入可学习参数
- **严格退化保证**：η=0 时数学等价于 π-StepNFT，不可能比 baseline 更差
- **梯度方向保持**：sign(y_i) = sign(y)，只调节幅度不改变方向

> **与 FEA 的关键区别**：FEA 用 embedding 做 ridge 回归 → TD advantage（可能改变标签符号 → softplus 崩溃）。DECO 用 velocity 偏差做距离度量 → 权重（永远不改变标签符号 → softplus 安全）。

## 2. 数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Rollout Worker                                │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ π_θ_old rollout — 独立 NFT SDE 去噪分支 (不变)            │           │
│  │                                                            │           │
│  │  → nft_xt, nft_v, nft_xnext, nft_step_index              │           │
│  │  → vlm_embedding (备用)                                    │           │
│  └──────────────────────────────────────────────────────────┘           │
│        │                                                                 │
│        ▼  Rollout 完成后                                                 │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ 【DECO 新增】_compute_deco_ref_deviations()               │           │
│  │                                                            │           │
│  │  cpu_weight_swap → 加载冻结 SFT 参考模型权重               │           │
│  │  对每个 env step:                                          │           │
│  │    v_ref = pi_ref(nft_xt, t, observation)  (no_grad)      │           │
│  │    D_i = mean(||v_ref - nft_v||²)  → 标量/env step        │           │
│  │  存入 forward_inputs["deco_ref_deviation"]                 │           │
│  └──────────────────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ���  send to actor
┌──────────────────────────────────────────────────────────────────────────┐
│                           Actor Worker                                   │
│                                                                          │
│  ┌─── Credit Assignment (DECO) ────────────────────────────────────┐    │
│  │  episode_rewards: 0/1 binary reward                              │    │
│  │  ref_deviations: D_i from rollout [n_steps, batch]               │    │
│  │                                                                  │    │
│  │  D_mean = mean(D_i),  D_std = std(D_i)                         │    │
│  │  w_i = sigmoid((D_i - D_mean) / D_std)       ∈ (0, 1)         │    │
│  │  y_i = (2r - 1) * (1 + η * w_i) * adv_clip_max                │    │
│  │                                                                  │    │
│  │  η=0 → y_i = (2r-1) * adv_clip_max  (= terminal_binary)       │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│        │                                                                │
│        ▼                                                                │
│  ┌─── Training (完全复用 NFT，不变) ─────────────────────────────┐    │
│  │  Mirror Construction → Mahalanobis Energy → softplus loss       │    │
│  │  唯一变化: advantages 是步级而非 episode 级                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
��──────────────────────────────────────────────────────────────────────────┘
```

## 3. DECO vs π-StepNFT：关键差异

| 特性 | π-StepNFT (terminal_binary) | DECO |
|------|----------------------------|------|
| 标签 y | 标量 $2r-1$，episode 级 | 向量 $y_i$，步级 |
| 信用分配 | 无（所有步等权） | 有（高偏差步获更大权重） |
| 额外组件 | 无 | 一次冻结 ref 模型前向传播 |
| 额外超参 | 无 | η（一个标量） |
| 额外计算 | 无 | ~25% rollout 开销 |
| η=0 时 | — | **严格退化为 π-StepNFT** |
| 可能比 baseline 差？ | — | **不可能**（η sweep 含 0） |

## 4. 时间约定

与 FlowIPO / π-StepNFT 一致，代码使用 t (t=0→clean, t=1→noise)：

| 概念 | 代码 (t) | 说明 |
|------|---------|------|
| Clean data | t = 0 | x₀ = action |
| Pure noise | t = 1 | x₁ = ε ~ N(0,I) |
| Schedule | linspace(1, 0, K+1) | 从噪声到干净 |
| SDE 步进 | x_{k+1} = flow_mean + σ·z | Euler-Maruyama |

## 5. 关键实现细节

### 5.1 冻结参考模型权重管理

DECO 使用 SFT checkpoint 作为冻结参考模型。权重在第一次 `sync_model_from_actor()` 时捕获，之后**永不更新**（区别于 FlowIPO/SAR 的 EMA 更新）。

```python
# huggingface_worker.py, sync_model_from_actor() 内部
# DECO: capture frozen reference weights on first sync only (SFT checkpoint)
if self._needs_deco_ref and self._deco_ref_weights_cpu is None:
    self._deco_ref_weights_cpu = {
        k: v.clone().cpu() for k, v in param_state_dict.items()
    }
# 注意: 没有 else 分支 → 后续 sync 不更新 ref 权重
```

**为什么冻结而非 EMA**：
- EMA 参考会随训练漂移，偏差信号变成"最近几步的变化"而非"从初始策略起的总变化"
- 冻结参考 = 固定锚点，D_i 的语义清晰：当前策略在这一步偏离 SFT 多少

### 5.2 参考偏差计算

使用 `cpu_weight_swap` 临时加载冻结 SFT 权重，复用 NFT 已保存的快照数据 (nft_xt, nft_v, nft_step_index)。

```python
# huggingface_worker.py, _compute_deco_ref_deviations() 内部
with cpu_weight_swap(self.hf_model, self._deco_ref_weights_cpu, ...):
    for fi in rollout_result.forward_inputs:
        # 复用 NFT 快照: x_t, step_index → 构造 timestep
        t = schedule[step_idx]
        # 冻结参考模型前向 (no_grad, 半精度)
        v_ref = self.hf_model.forward_velocity(None, x_t, t, observation=obs)
        # 偏差: D_i = mean(||v_ref - v_old||^2)
        D_i = ((v_ref_crop - v_old) ** 2).mean(dim=(-2, -1))
        fi["deco_ref_deviation"] = D_i.cpu()
```

**计算开销**：
- 每个 env step：1 次 VLM prefix forward + 1 次 suffix forward（参考模型）
- 所有 env step 共享一次 weight swap（O(模型参数数) memcpy）
- 总额外开销约 **20-25%** rollout 时间

### 5.3 步级标签构造

在 actor worker 的 credit assignment 阶段，从 D_i 构造步级标签。

```python
# credit_assignment.py, compute_deco_advantages() 核心逻���
D_mean = ref_deviations.mean()
D_std = ref_deviations.std() + 1e-8
w = sigmoid((D_i - D_mean) / D_std)          # ∈ (0, 1)
y_i = (2r - 1) * (1 + η * w) * adv_clip_max  # 步级标签
```

**Sigmoid 归一化的优势**（对比线性归一化）：
- 自动压缩极端值，防止异常偏差主导
- 输出有界 (0, 1)，标签 |y_i| ∈ [1, 1+η]
- batch 归一化确保跨 epoch 的相对排序一致

### 5.4 与 softplus 对比 loss 的兼容性

**DECO 安全性保证**：
1. `sign(y_i) = sign(2r-1)` 恒成立（因为 `1 + η * w_i > 0`）
2. → 标签方向永远来自 terminal binary → 永远正确
3. → softplus 不对称性不会造成 damage
4. → 与 FEA 的关键区别：FEA 可能产生错误符号，DECO 不会

## 6. 修改的文件一览

| 文件 | 修改内容 |
|------|---------|
| `rlinf/algorithms/credit_assignment.py` | 新增 `compute_deco_advantages()` — sigmoid 归一化偏差权重 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 新增 `_needs_deco_ref` 标志 + 冻结权重管理 + `_compute_deco_ref_deviations()` |
| `rlinf/workers/actor/fsdp_actor_worker.py` | 新增 `deco_eta` 配置 + `"deco"` credit assignment 分支 + DECO metrics |
| `examples/embodiment/config/libero_object_deco_openpi_quickstart.yaml` | DECO YAML 配置 |

**未修改的文件**（完全复用 NFT 基础设施）：
- `rlinf/algorithms/losses.py` — `compute_flow_nft_loss()` 不变（只改了输入 advantages）
- `rlinf/algorithms/registry.py` — `"flow_nft"` bypass 复用
- `rlinf/models/embodiment/openpi/openpi_action_model.py` — NFT 快照收集不变
- `rlinf/workers/actor/fsdp_actor_worker.py` — NFT 训练分支不变（loss 计算不变）

## 7. 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `flow_nft_adv_type` | `"deco"` | 设为 `"deco"` 启用 DECO |
| `flow_nft_deco_eta` | 2.0 | **唯一新超参**。偏差调制强度 |
| | | η=0 → 退化为 terminal_binary |
| | | η=2 → max weight ratio ≈ 3× (推荐起点) |
| | | sweep: {0, 1, 2, 3, 5} |
| `flow_nft_beta` | 1.0 | 镜像构造缩放 β（不变） |
| `flow_nft_kl_beta` | 0.0001 | KL trust region 权重（不变） |
| `flow_nft_max_drift` | 0.5 | 速度变化裁剪（不变） |
| `flow_nft_dpo_beta` | 1.0 | softplus logit 缩放（不变） |
| `flow_nft_noise_level` | 0.2 | SDE 探索噪声（不变） |
| `flow_nft_adv_clip_max` | 1.0 | Advantage 裁剪（不变） |

**调参建议**：
- `deco_eta`：唯一需要调的新参数。从 2.0 开始：
  - Long 任务不够好 → 增到 3 或 5（更强的步级区分）
  - 训练不稳定 → 减到 1（更保守）
  - 与 baseline 无差异 → 检查 D_i 分布（可能偏差均匀，假设不成立）
- 其他参数：保持与 π-StepNFT 一致，不需要调

## 8. 监控指标

在 NFT 原有指标基础上，DECO 新增以下 TensorBoard 指标：

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `deco/D_mean` | 参考偏差均值 | 随训练逐步增大（策略偏离 SFT） |
| `deco/D_std` | 参考偏差标准差 | 非零（步间有区分度） |
| `deco/w_mean` | Sigmoid 权重均值 | ≈ 0.5（归一化后） |
| `deco/w_std` | Sigmoid 权重标准差 | 非零（越大说明区分度越好） |
| `deco/w_min` / `w_max` | 权重范围 | min > 0, max < 1 |
| `deco/y_abs_mean` | 步级标签绝对值均值 | ∈ [adv_clip, (1+η)*adv_clip] |
| `deco/y_abs_max` | 步级标签绝对值最大 | ≤ (1+η)*adv_clip |
| `deco/eta` | 当前 η 值 | 固定（除非使用自适应调度） |
| `nft/adv_type` | 3.0 = DECO | 固定 3.0 |

**复用的 NFT 指标**（含义和期望趋势不变）：
- `nft_loss`, `kl_loss`, `total_loss`, `pref_acc`, `logit_mean`, `delta_E_mean` 等
- `env/success_once`, `env/episode_reward`

**异常诊断**：
- `deco/D_std ≈ 0` → 所有步偏差均匀，DECO 退化为 terminal_binary（假设不成立）
- `deco/w_std ≈ 0` → 同上
- `deco/D_mean` 不增长 → 策略没有偏离 SFT，可能 lr 太低或 RL 信号太弱
- `nft_loss` 不下降 → 检查 NFT 基础组件（与 DECO 无关）

## 9. 一键启停

```bash
# 训练
bash run_embodiment.sh libero_object_deco_openpi_quickstart

# 评估
bash run_embodiment.sh libero_object_deco_openpi_quickstart --only_eval

# 恢复训练
bash run_embodiment.sh libero_object_deco_openpi_quickstart --resume_dir <path>

# 对比实验: η sweep
for eta in 0 1 2 3 5; do
  bash run_embodiment.sh libero_object_deco_openpi_quickstart \
    algorithm.flow_nft_deco_eta=$eta \
    runner.logger.experiment_name="deco_eta${eta}"
done
```

## 10. 关键公式速查

```
=== DECO 新增 (Phase 1 + Phase 2) ===

Reference Deviation (rollout worker):
    v_ref = pi_ref(x_t, t, s_i)                    (frozen SFT model, no_grad)
    D_i = mean(||v_ref - v_old||²)                  (per env step scalar)

Step-level Label (actor worker):
    D_mean = mean(D_i),  D_std = std(D_i) + 1e-8
    w_i = sigmoid((D_i - D_mean) / D_std)           ∈ (0, 1)
    y_i = (2r - 1) * (1 + η * w_i) * adv_clip_max  (step-level)

=== 以下完全复用 NFT (Phase 3, 不变) ===

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

NFT Loss:         y = clip(advantage / c, -1, 1)       ← advantage 现在是步级的
                  logit = ½ · y · (E⁺ - E⁻)
                  L_NFT = softplus(logit)
                  L_KL = ||v_θ - v_old||²
                  L = L_NFT + kl_beta · L_KL
```

## 11. 理论保证

### 11.1 梯度方向保持

DECO 的梯度与 π-StepNFT 方向一致：

```
∇_θ L_DECO ∝ σ(z_t) · |y_i| · sign(y) · (∂v_θ/∂θ)^T · B_t · Σ_t^{-1} · e_t
```

由于 `sign(y_i) = sign(2r-1) = sign(y)`（因为 `1 + η·w_i > 0` 恒成立），梯度方向与 π-StepNFT 的 oracle 对齐性完全一致。DECO 不改变"往哪走"，只改变"每步走多快"。

### 11.2 有界性

$|y_i| \in [1, 1+\eta]$（因为 $w_i \in (0,1)$）。不存在梯度爆炸或消失风险。

### 11.3 严格退化

$\eta = 0 \Rightarrow y_i = (2r-1) \cdot 1 = y$。DECO 严格退化为 π-StepNFT。

### 11.4 信噪比改善

在"关键步假设"下（关键步 $\leftrightarrow$ 高偏差步），DECO 的有效梯度 SNR 高于 π-StepNFT：

```
SNR_DECO = (|K| · w̄_K) / (|N| · w̄_N) · SNR_per_step > |K|/|N| · SNR_per_step = SNR_NFT
```

当假设不成立时，w_i 趋向均匀，DECO 退化为 π-StepNFT（不会变差）。

## 12. 风险与已知限制

| 风险 | 严重性 | 缓解措施 |
|------|--------|---------|
| "高偏差步=关键步"假设不成立 | 中 | 退化为 terminal_binary（不变差） |
| 偏差随步号单调增长（累积漂移） | 中 | Batch 归一化 + Sigmoid 压缩 |
| 冻结 ref 模型长期不更新 | 低-中 | Batch 归一化保证相对排序有效 |
| η 调参敏感 | 低 | Sigmoid 天然限制范围，η=0 退化 |
| 额外 rollout 计算开销 (~25%) | 低 | no_grad + 半精度推理 |
