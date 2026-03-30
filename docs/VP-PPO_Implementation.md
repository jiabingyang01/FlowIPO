# VP-PPO (VLM-Potential PPO) — 实现文档

## 1. 算法概述

VP-PPO 的核心思想：**利用冻结 VLM 的特征作为 PBRS 势函数，给 PPO 提供 dense reward shaping**。

```
πRL PPO:   Rollout → [r_env]                         → Critic → GAE → PPO Update
VP-PPO:    Rollout → [r_env + α·(γΦ(s') - Φ(s))]   → Critic → GAE → PPO Update
                            ↑ 唯一修改点
```

关键创新：**冻结 VLM 天然满足 PBRS 势函数条件**（Φ 不依赖 θ）：
- Φ(s) = Σ_l w_l · cos_sim(z_t^(l), z_success^(l))
- PBRS 定理保证：shaped reward 不改变最优策略（无 reward hacking 风险）
- 多层自适应权重：浅层编码空间距离，深层编码语义进度，自动选择区分度最大的层
- VLM forward pass 已执行（为 Action Expert 提供条件），特征提取零额外推理成本

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
│  │ sample_actions() 中截取 VLM 特征 (零额外成本):               │       │
│  │   prefix_output = VLM(images, language)     [已有]           │       │
│  │   vlm_embedding = mean_pool(prefix_output)  [新增: 存入结果] │       │
│  └──────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  send to actor
┌──────────────────────────────────────────────────────────────────────────┐
│                           Actor Worker                                   │
│                                                                          │
│  ┌─── PBRS Reward Shaping (新增) ────────────────────────────────┐     │
│  │  1. 从 rollout 取 vlm_embedding [n_steps+1, batch, hidden]    │     │
│  │  2. 更新 SuccessFeatureBuffer (成功轨迹末帧 EMA)              │     │
│  │  3. 计算势函数 Φ_t = cos_sim(emb_t, target)                   │     │
│  │  4. PBRS: r'_t = r_t + α · (γ·Φ_{t+1} - Φ_t)               │     │
│  │  5. 用 r' 替换 rollout_batch["rewards"]                       │     │
│  └────────────────────────────────────────────────────────────────┘     │
│        │                                                                │
│        ▼                                                                │
│  ┌─── 标准 PPO Credit Assignment (完全不变) ─────────────────────┐     │
│  │  advantages, returns = GAE(r', values, dones, γ, λ)           │     │
│  └────────────────────────────────────────────────────────────────┘     │
│        │                                                                │
│        ▼                                                                │
│  ┌─── 标准 PPO Training (完全不变) ──────────────────────────────┐     │
│  │  ratio = π_new / π_old                                         │     │
│  │  L_clip = -min(ratio·A, clip(ratio)·A)                         │     │
│  │  L_V = ||V_θ - returns||²                                      │     │
│  │  L = L_clip + c_v · L_V                                        │     │
│  └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
```

**设计原则：VP-PPO 不修改任何 PPO 逻辑，只在 reward 进入 GAE 之前做一步 shaping。**

## 3. 为什么 VP-PPO 而不是直接加 dense reward

| 方案 | 问题 |
|------|------|
| 直接加 VLM cosine similarity 作为 reward | 改变最优策略（reward hacking 风险） |
| 训练 Reward Model | 额外计算 + 额外训练 + 分布漂移 |
| VP-PPO (PBRS) | **数学保证不改变最优策略 + 零额外训练 + 零额外推理** |

PBRS 的 telescope 性质：沿整条轨迹求和，shaping reward 相消为 `γ^T Φ(s_T) - Φ(s_0)`，只影响初末态，中间步的噪声被消除。

## 4. 势函数设计

### 4.1 Phase 1：单层（最小实现，推荐先验证）

直接复用现有 `vlm_embedding`（final layer mean pool）：

```python
# 势函数
Φ(s_t) = cos_sim(vlm_embedding_t, success_target)

# PBRS reward
r'_t = r_t + α · (γ · Φ(s_{t+1}) - Φ(s_t))
```

**优势**：零代码修改 VLM 模型，仅需在 PPO 路径中也提取 vlm_embedding（NFT 路径已有）。

### 4.2 Phase 2：多层自适应（如果 Phase 1 有效）

需要修改 `paligemma_with_expert.forward()` 以返回中间层 hidden states：

```python
# 选取 4 个层
layers = [4, 12, 20, 24]  # 浅层→深层

# 每层势函数
Φ_l(s_t) = cos_sim(z_t^(l), z_success^(l))

# 自适应层权重（在线计算，按成功/失败终态区分度）
Δ_l = E_success[Φ_l(s_final)] - E_fail[Φ_l(s_final)]
w_l = max(Δ_l, 0) / (Σ max(Δ, 0) + ε)

# 组合势函数
Φ(s_t) = Σ_l w_l · Φ_l(s_t)
```

多层互补的关键价值：浅层（空间距离）和深层（语义进度）在任务不同阶段互补，避免单层的非单调性问题。

### 4.3 Success Feature Buffer

```python
class SuccessFeatureBuffer:
    """按任务维护成功终态 VLM 特征的 EMA 目标。"""

    def __init__(self, ema_rate=0.99, max_size=500):
        self.ema_target = {}      # task_hash → Tensor [hidden_dim]
        self.ema_rate = ema_rate
        self.count = {}           # task_hash → int (累计成功数)

    def update(self, task_hash, success_final_embs):
        """用成功轨迹末帧特征更新 EMA 目标。

        Args:
            task_hash: 任务标识（用 lang token hash 代替 task_id）
            success_final_embs: [n_success, hidden_dim]
        """
        batch_mean = success_final_embs.mean(dim=0)
        if task_hash not in self.ema_target:
            self.ema_target[task_hash] = batch_mean
            self.count[task_hash] = success_final_embs.shape[0]
        else:
            self.ema_target[task_hash] = (
                self.ema_rate * self.ema_target[task_hash]
                + (1 - self.ema_rate) * batch_mean
            )
            self.count[task_hash] += success_final_embs.shape[0]

    def get_target(self, task_hash):
        """返回 EMA 目标特征，或 None（冷启动）。"""
        return self.ema_target.get(task_hash, None)

    def has_enough(self, task_hash, min_count=5):
        return self.count.get(task_hash, 0) >= min_count
```

**任务标识**：当前 rollout pipeline 没有显式 `task_id`。用 language token 的 hash 值代替（同一任务的 instruction 相同）。对于单任务训练（如 LIBERO-Object quickstart），所有 episode 共用一个 hash，buffer 退化为全局 EMA。

### 4.4 冷启动 Fallback

训练最初期（buffer 为空或样本不足），VP-PPO 不做 shaping，退化为标准 PPO：

```python
if not buffer.has_enough(task_hash, min_count=5):
    return env_reward  # 原始 reward，不做 shaping
```

保守策略：至少 5 个成功 episode 后才开始 PBRS。避免用不准的 target 误导 Critic。

## 5. 代码集成点

### 5.1 VLM Embedding 提取（Rollout 侧）

**文件**：`rlinf/models/embodiment/openpi/openpi_action_model.py`

**当前状态**：`vlm_embedding` 仅在 NFT 路径提取（line 492-493）。PPO 的 normal denoising path（line 550+）不提取。

**修改**：在 normal path 的 `result` dict（line 620-628）中也加入 vlm_embedding。

```python
# === 修改 sample_actions() normal path (line ~490) ===
# 现有: 只有 use_nft 时提取
# 修改: PPO (use_vlm_value) 也提取
vlm_embedding = None
if use_nft or self.use_vlm_value:  # 新增 self.use_vlm_value 条件
    vlm_embedding = prefix_output.mean(dim=1).detach()

# === 修改 normal path result dict (line ~620) ===
result = {
    "actions": x_0,
    "chains": chains,
    "prev_logprobs": log_probs,
    "prev_values": values,
    "denoise_inds": denoise_inds,
}
if vlm_embedding is not None:
    result["vlm_embedding"] = vlm_embedding  # 新增
```

**关键**：`prefix_output` 在 normal path 已计算（line 457），只需 `.mean(dim=1).detach()`，无额外 VLM forward。

### 5.2 VLM Embedding 传递（Rollout Worker 侧）

**文件**：`rlinf/workers/rollout/hf/huggingface_worker.py`（或等效 rollout worker）

**修改**：确保 `vlm_embedding` 从 `sample_actions()` 的返回值传入 `forward_inputs`。

当前 NFT 已有传递逻辑（openpi_action_model.py line 416-418）：
```python
for _nft_key in ("nft_xt", "nft_v", "nft_xnext", "nft_step_index",
                  "nft_noise_level", "vlm_embedding"):
    if _nft_key in outputs:
        forward_inputs[_nft_key] = outputs[_nft_key]
```

需要确保 PPO 路径也执行类似传递。如果现有框架已将 `sample_actions()` 的所有输出传入 `forward_inputs`，则无需修改。否则需要添加 `vlm_embedding` 到传递列表。

### 5.3 PBRS Reward Shaping（Actor Worker 侧 — 核心修改）

**文件**：`rlinf/workers/actor/fsdp_actor_worker.py`

**插入点**：`compute_advantages_and_returns()` 方法（line 1275），在标准 PPO 路径的 GAE 计算之前。

```python
def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
    # ... 现有 flow RL 分支 ...
    if self._is_flow_nft:
        return self._compute_flow_nft_credit_assignment()
    # ... 其他 flow 分支 ...

    # ========== VP-PPO: PBRS Reward Shaping (新增) ==========
    if self._use_vp_ppo:
        self._apply_vlm_potential_shaping()
    # ========================================================

    # 标准 PPO GAE 计算（完全不变）
    kwargs = {
        "rewards": self.rollout_batch["rewards"],  # 已被 shaping 修改
        "dones": self.rollout_batch["dones"],
        "values": self.rollout_batch.get("prev_values", None),
        "gamma": self.cfg.algorithm.get("gamma", 1),
        "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
        ...
    }
    advantages_and_returns = calculate_adv_and_returns(**kwargs)
    ...
```

### 5.4 `_apply_vlm_potential_shaping()` 实现

```python
def _apply_vlm_potential_shaping(self):
    """VP-PPO: 用冻结 VLM 特征构造 PBRS shaped reward，原地修改 rollout_batch['rewards']。"""
    rewards = self.rollout_batch["rewards"]          # [n_steps, batch, chunk]
    forward_inputs = self.rollout_batch.get("forward_inputs", {})
    vlm_embs = forward_inputs.get("vlm_embedding", None)  # [n_steps*batch, hidden] or [n_steps, batch, hidden]

    if vlm_embs is None:
        return  # 没有 VLM embedding，跳过 shaping

    n_steps = rewards.shape[0]
    batch_size = rewards.shape[1]

    # 整形为 [n_steps, batch, hidden_dim]
    if vlm_embs.dim() == 2:
        hidden_dim = vlm_embs.shape[-1]
        vlm_embs = vlm_embs.reshape(n_steps, batch_size, hidden_dim)

    # Episode-level reward (判断成功/失败)
    loss_mask = self.rollout_batch.get("loss_mask", None)
    if loss_mask is not None:
        episode_rewards = (rewards * loss_mask).sum(dim=(0, 2))
    else:
        episode_rewards = rewards.sum(dim=(0, 2))
    episode_rewards = episode_rewards.clamp(0, 1)  # [batch]
    success_mask = episode_rewards > 0.5  # [batch]

    # --- 更新 Success Feature Buffer ---
    # 任务标识: 单任务训练用全局 hash; 多任务用 language token hash
    task_hash = 0  # 简化: 单任务场景
    if success_mask.any():
        # 成功轨迹的最后一步 embedding
        success_final_embs = vlm_embs[-1, success_mask, :]  # [n_success, hidden]
        self._vp_buffer.update(task_hash, success_final_embs.detach())

    # --- 获取目标特征 ---
    target = self._vp_buffer.get_target(task_hash)
    if target is None or not self._vp_buffer.has_enough(task_hash):
        return  # 冷启动，不做 shaping

    target = target.to(vlm_embs.device)  # [hidden_dim]

    # --- 计算势函数 Φ_t ---
    # vlm_embs: [n_steps, batch, hidden], target: [hidden]
    phi = torch.nn.functional.cosine_similarity(
        vlm_embs, target.unsqueeze(0).unsqueeze(0).expand_as(vlm_embs),
        dim=-1
    )  # [n_steps, batch]

    # --- PBRS reward: r' = r + α · (γ·Φ_{t+1} - Φ_t) ---
    gamma = self.cfg.algorithm.get("gamma", 0.99)
    alpha = self._vp_cfg["alpha"]

    # 线性衰减 alpha
    if self._vp_cfg["alpha_decay"]:
        warmup_iters = self._vp_cfg["alpha_warmup_iters"]
        alpha_min = self._vp_cfg["alpha_min"]
        progress = min(self.optimizer_steps / max(warmup_iters, 1), 1.0)
        alpha = alpha * (1.0 - progress) + alpha_min * progress

    # Φ_{t+1} - Φ_t (差分)
    # 注意: phi 是 [n_steps, batch]，需要 Φ_{n_steps}（最后一步之后）
    # 用最后一步的 Φ 作为 Φ_{T}（terminal state 没有额外 embedding）
    phi_next = torch.cat([phi[1:], phi[-1:]], dim=0)  # [n_steps, batch]
    pbrs = alpha * (gamma * phi_next - phi)  # [n_steps, batch]

    # 加到 reward 上 (broadcast 到 chunk 维度)
    if rewards.dim() == 3:
        # rewards: [n_steps, batch, chunk], pbrs: [n_steps, batch]
        # 只加到第一个 chunk 维度（避免重复计算）
        pbrs_expanded = torch.zeros_like(rewards)
        pbrs_expanded[:, :, 0] = pbrs
        self.rollout_batch["rewards"] = rewards + pbrs_expanded
    else:
        self.rollout_batch["rewards"] = rewards + pbrs.unsqueeze(-1)

    # 记录 metrics
    self._vp_metrics = {
        "vp_ppo/phi_mean": phi.mean().item(),
        "vp_ppo/phi_std": phi.std().item(),
        "vp_ppo/pbrs_mean": pbrs.mean().item(),
        "vp_ppo/pbrs_abs_mean": pbrs.abs().mean().item(),
        "vp_ppo/alpha": alpha,
        "vp_ppo/success_in_buffer": self._vp_buffer.count.get(task_hash, 0),
        "vp_ppo/phi_success_final": phi[-1, success_mask].mean().item() if success_mask.any() else 0.0,
        "vp_ppo/phi_fail_final": phi[-1, ~success_mask].mean().item() if (~success_mask).any() else 0.0,
    }
```

### 5.5 初始化（Actor Worker `__init__` 或 setup）

```python
# 在 fsdp_actor_worker.py 的初始化中添加:
self._use_vp_ppo = self.cfg.algorithm.get("use_vp_ppo", False)
if self._use_vp_ppo:
    self._vp_cfg = {
        "alpha": self.cfg.algorithm.get("vp_ppo_alpha", 0.2),
        "alpha_min": self.cfg.algorithm.get("vp_ppo_alpha_min", 0.05),
        "alpha_decay": self.cfg.algorithm.get("vp_ppo_alpha_decay", True),
        "alpha_warmup_iters": self.cfg.algorithm.get("vp_ppo_alpha_warmup_iters", 200),
        "buffer_ema_rate": self.cfg.algorithm.get("vp_ppo_buffer_ema_rate", 0.99),
        "buffer_min_count": self.cfg.algorithm.get("vp_ppo_buffer_min_count", 5),
    }
    self._vp_buffer = SuccessFeatureBuffer(
        ema_rate=self._vp_cfg["buffer_ema_rate"],
    )
    self._vp_metrics = {}
```

## 6. 修改的文件（已实现 ✅）

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `rlinf/algorithms/reward_shaping.py` | **新文件**：`SuccessFeatureBuffer` + `compute_vlm_potential()` + `compute_pbrs_reward()` | ✅ |
| `rlinf/workers/actor/fsdp_actor_worker.py` | VP-PPO 初始化 + `_apply_vlm_potential_shaping()` + GAE 前调用 + metrics | ✅ |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | 新增 `collect_vlm_embedding` config + PPO normal path 也提取 vlm_embedding | ✅ |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | VP-PPO 时设置 `collect_vlm_embedding = True` | ✅ |
| `examples/embodiment/config/libero_object_vpppo_openpi_quickstart.yaml` | VP-PPO YAML 配置（基于 PPO quickstart） | ✅ |

**复用现有基础设施**（无需修改）：
- `advantages.py` — GAE 计算完全不变
- `losses.py` — PPO clip loss 完全不变
- PPO training branch — 完全不变（reward 已被 shaping）
- `predict_action_batch()` — vlm_embedding 已在 NFT key 传递循环中

## 7. YAML 配置

基于 `libero_object_ppo_openpi_quickstart.yaml`，新增 VP-PPO 参数：

```yaml
# === libero_object_vpppo_openpi_quickstart.yaml ===
# 继承标准 PPO 配置，仅新增 VP-PPO 部分

algorithm:
  # --- 标准 PPO 参数（与 quickstart 完全一致） ---
  adv_type: gae
  loss_type: actor_critic
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  update_epoch: 4
  # ...（其余同 libero_object_ppo_openpi_quickstart.yaml）

  # --- VP-PPO 新增参数 ---
  use_vp_ppo: True                # 开关：True=VP-PPO, False=标准 PPO
  vp_ppo_alpha: 0.2               # PBRS 基础缩放系数
  vp_ppo_alpha_min: 0.05          # PBRS 最小缩放（衰减下限）
  vp_ppo_alpha_decay: True        # 是否线性衰减 alpha
  vp_ppo_alpha_warmup_iters: 200  # alpha 衰减到 alpha_min 的步数
  vp_ppo_buffer_ema_rate: 0.99    # 成功特征 EMA 更新率
  vp_ppo_buffer_min_count: 5      # 冷启动阈值（至少 N 个成功 episode 后才 shaping）
```

**一键对比**：`use_vp_ppo: False` 即退化为标准 PPO，零代码差异。

## 8. 超参数

| 参数 | 默认值 | 说明 | 调参建议 |
|------|-------|------|---------|
| `vp_ppo_alpha` | 0.2 | PBRS 缩放系数。控制 shaping reward 相对于 env reward 的强度 | 过大 → Critic 被 shaping 主导；过小 → 无效果。[0.1, 0.5] |
| `vp_ppo_alpha_min` | 0.05 | 衰减下限 | 不要设为 0（保留少量 OOD 鲁棒性信号） |
| `vp_ppo_alpha_decay` | True | 是否线性衰减 | True=Critic 学好后逐渐减弱 shaping |
| `vp_ppo_alpha_warmup_iters` | 200 | 衰减到 alpha_min 的步数 | ≈ 训练总步数的 1/3~1/2 |
| `vp_ppo_buffer_ema_rate` | 0.99 | 成功特征 EMA 率 | 越大 → 目标更稳定；越小 → 跟踪策略变化更快 |
| `vp_ppo_buffer_min_count` | 5 | 冷启动阈值 | 低于此数不做 shaping。设太高 → shaping 启动慢 |
| `gamma` | 0.99 | 折扣因子 | 与标准 PPO 一致，VP-PPO 复用 |

## 9. PBRS 的数学安全性

### 9.1 策略不变性

**PBRS 定理 (Ng et al., 1999)**：若 Φ(s) 仅依赖状态 s，则 shaped reward `r' = r + γΦ(s') - Φ(s)` 不改变最优策略。

**VP-PPO 满足前提**：
- Φ(s) = cos_sim(VLM(o), target)
- VLM 冻结 → VLM(o) 不依赖 θ
- target 在每个 PPO update 内为常量（EMA 只在 rollout 间更新）

### 9.2 有界性

- cos_sim ∈ [-1, 1] → Φ ∈ [-1, 1]
- |PBRS| = |α · (γΦ(s') - Φ(s))| ≤ α · (1 + γ) ≤ 2α = 0.4
- 与 env reward（0 或 1）量级相当但略小 → 辅助而非替代

### 9.3 最差情况退化

若 VLM 特征不编码任务进度（Φ 随机波动）：
- PBRS reward 在长轨迹上 telescope 相消 → 零均值随机噪声
- Critic 通过 TD learning 自然忽略零均值噪声
- VP-PPO 退化为标准 PPO（不会比标准 PPO 更差）

## 10. 与现有方法的对比

| 特性 | πRL PPO | FEA (NFT variant) | VP-PPO |
|------|---------|-------------------|--------|
| VLM 特征用途 | 无 | Advantage 幅度调节 | **PBRS 势函数** |
| 修改 reward | 否 | 否 | **是（shaped reward）** |
| 修改 advantage | 否 | 是（label 幅度） | **否** |
| 修改 loss | 否 | 否 | **否** |
| 理论保证 | N/A | 无 | **PBRS 策略不变性** |
| 最差情况 | baseline | 比 ±1 差 | **退化为 PPO** |
| 与 Critic 的关系 | 独立 | 替代 | **辅助（early hint）** |

**VP-PPO vs FEA 的根本区别**：
- FEA 把 VLM embedding 放进 softplus contrastive loss 的 **label** → 对噪声极度敏感 → 失败
- VP-PPO 把 VLM embedding 放进 **reward**（GAE 之前）→ Critic 通过 TD learning 平滑噪声 → 鲁棒

## 11. 监控指标

TensorBoard 中关注：

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `vp_ppo/phi_mean` | 全局平均势函数值 | 上升（策略改善 → 状态更接近成功） |
| `vp_ppo/phi_success_final` | 成功轨迹末帧 Φ | 接近 1.0 |
| `vp_ppo/phi_fail_final` | 失败轨迹末帧 Φ | 明显低于成功 |
| `vp_ppo/pbrs_mean` | PBRS reward 均值 | ≈ 0（PBRS 理论上零均值） |
| `vp_ppo/pbrs_abs_mean` | PBRS reward 绝对值均值 | 0.01~0.1（有信号但不过大） |
| `vp_ppo/alpha` | 当前 α 值 | 线性衰减 |
| `vp_ppo/success_in_buffer` | Buffer 累计成功数 | 单调增长 |
| 标准 PPO 指标 | success_rate, reward, value_loss 等 | 与 PPO baseline 对比 |

**关键验证信号**：`phi_success_final - phi_fail_final > 0.1` 说明 VLM 特征确实区分了成功/失败 → PBRS 信号有效。若差值 < 0.05 → VLM 特征无区分度 → VP-PPO 无效。

## 12. 验证计划

### Phase 0：离线验证（0.5 天）— 必须先做

不需要训练。用 SFT 模型跑 50 个 episode，记录每步的 vlm_embedding，离线计算 Φ 曲线：

```python
# 验证脚本伪代码
for episode in rollout_episodes:
    for t in range(T):
        phi_t = cos_sim(vlm_embs[t], success_target)
    plot(phi_t)  # 成功轨迹 → 应单调上升; 失败 → 应平坦/下降
```

**Pass 标准**：成功轨迹的 Φ 曲线整体上升趋势明显（不要求严格单调）。

### Phase 1：单层 VP-PPO on LIBERO-Object（1 天）

- 对比：VP-PPO vs πRL PPO（同配置，仅 `use_vp_ppo` 开/关）
- 重点观察前 100 iter 的 Critic warm-up 速度
- 预期：warm-up 加速 10-30%，最终 success rate 持平或小幅提升

### Phase 2：OOD 泛化（如果 Phase 1 有效）

- ManiSkill Semantic OOD 测试
- 预期：VP-PPO OOD 好于 PPO（VLM 特征的预训练鲁棒性）

## 13. 关键公式速查

```
势函数:    Φ(s_t) = cos_sim(VLM_embed(o_t), EMA_success_target)

PBRS:      r'_t = r_t + α · (γ · Φ(s_{t+1}) - Φ(s_t))

α衰减:    α_m = max(α_0 · (1 - m/M_warmup), α_min)

GAE:       A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
           δ_t = r'_t + γ · V(s_{t+1}) - V(s_t)   ← 注意这里用 r'

PPO clip:  L = -min(ratio · A, clip(ratio, 1±ε) · A)   ← 完全不变
```

## 14. 一键启停

```bash
# VP-PPO 训练
bash run_embodiment.sh libero_object_vpppo_openpi_quickstart

# 标准 PPO 对比（同配置，关闭 VP-PPO）
# 只需修改 yaml: use_vp_ppo: False
bash run_embodiment.sh libero_object_ppo_openpi_quickstart

# 评估
bash run_embodiment.sh libero_object_vpppo_openpi_quickstart --only_eval
```
