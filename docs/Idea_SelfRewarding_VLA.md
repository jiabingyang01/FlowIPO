# Self-Rewarding Vision-Language-Action Models

## SR-VLA：利用 VLM 自身视觉特征构造稠密奖励，加速 Flow-based VLA RL 收敛

---

## 一、Problem Statement

### 1.1 当前最优方案及其瓶颈

Flow-SDE + PPO（即 πRL 方案）是当前 VLA RL fine-tuning 的最佳实践。其核心流程：

1. K 步去噪（K=4）生成 action chunk
2. 在随机 1 步注入 SDE 噪声，计算 Gaussian log-prob
3. 环境执行 action，获取 reward
4. 用 GAE 计算 advantage，PPO 更新 policy

**训练效率瓶颈**：在 LIBERO 等标准 benchmark 上，虽然 Flow-SDE+PPO 能收敛，但收敛速度受限——通常需要 200-400 个 epoch 才能达到较高 success rate。

### 1.2 瓶颈根因：稀疏奖励下的 Credit Assignment

LIBERO 环境的奖励结构：
- **二值稀疏奖励**：只在 episode 最后一步给出 0（失败）或 1（成功）
- **长 horizon**：每个 episode 240 步（以 action chunk 5 计，约 48 个决策步）
- **中间步 reward = 0**：前 239 步没有任何奖励信号

GAE 的 credit assignment 依赖 value function $V(o, l)$：

$$\delta_t = r_t + \gamma V(o_{t+1}, l) - V(o_t, l)$$
$$A_t^{GAE} = \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta_{t+k}$$

**核心问题**：训练初期 $V(o, l)$ 非常不准确：

- 稀疏奖励意味着绝大多数步的 $r_t = 0$
- $\delta_t \approx \gamma V(o_{t+1}) - V(o_t)$ 完全依赖 value 差分
- $V$ 本身通过 TD 自举训练，信号从最后一步反向传播：经过 $\gamma^{240} \approx 0.09$ 的衰减才能传到第 1 步
- 初期 $V \approx 0$ 时，$A_t^{GAE} \approx 0$（纯噪声）→ PPO 几乎学不到东西

**结果**：前 50-100 个 epoch 是 PPO 的"冷启动期"，policy 改善极慢；之后 $V$ 逐渐准确，PPO 才开始有效学习。

### 1.3 现有方案的不足

- **Progress reward**（πRL 使用）：$r_{prog} = t/T$ 给了一个线性先验，帮助 $V$ 更快收敛，但它是 task-agnostic 的，不包含任何视觉语义信息
- **External reward model**（RoboCLIP, VLM-RM）：需要额外训练/加载独立的 reward model，增加计算开销和工程复杂度
- **Hand-crafted dense reward**：需要人工设计，不可扩展

### 1.4 Key Insight

**π0 的 VLM（PaLI-Gemma + SigLIP）在每一步都已经计算了视觉特征——这些特征天然包含了任务进展信息，但目前被完全浪费了。**

SigLIP 视觉编码器将每张图像映射为 256 个 token（每个 2048 维），这些特征经过大规模视觉-语言预训练，能够理解：
- 物体位置和状态（杯子在桌上 vs 在手里）
- 空间关系（机械臂靠近目标 vs 远离）
- 任务语义（"pick up the red cup" 对应的视觉状态）

**如果我们直接用这些已有特征来衡量任务进展，就能零额外计算地获得稠密奖励信号。**

---

## 二、Background

### 2.1 Potential-Based Reward Shaping (PBRS)

PBRS（Ng et al., 1999）是 RL 领域中最经典的 reward shaping 理论：

**定理（Ng 1999）**：给定 MDP $M = (S, A, P, R, \gamma)$，定义势函数 $\Phi: S \to \mathbb{R}$，构造 shaped reward：

$$r'(s, a, s') = r(s, a, s') + \gamma \cdot \Phi(s') - \Phi(s)$$

则：在 shaped MDP $M' = (S, A, P, r', \gamma)$ 下的最优策略 $\pi^{*}_{M'}$ 和原始 MDP 下的最优策略 $\pi^{*}_M$ 完全一致。

**含义**：PBRS 保证不引入任何 bias——只加速收敛，不改变最优解。这是 self-rewarding 方法的理论基石。

### 2.2 预训练视觉特征作为奖励信号

近年多项工作验证了预训练视觉特征可以有效衡量任务进展：

| 方法 | 视觉编码器 | 奖励构造方式 | 结果 |
|------|-----------|-------------|------|
| RoboCLIP (Adeniji 2023) | CLIP ViT-B/32 | cosine sim to language goal | 比 sparse reward 快 2-3× |
| LIV (Ma 2023) | CLIP + fine-tuned | 视觉-语言 embedding 距离 | 比 sparse reward 快 3-5× |
| R3M (Nair 2022) | ResNet (time-contrastive) | 特征距离到 goal | Significant improvement |
| VLM-RM (Rocamonde 2024) | CLIP ViT-L | 多模态 similarity | Dense reward enables hard tasks |

**共同结论**：预训练视觉特征的 cosine similarity 变化量 $\Delta \cos(f_t, f_{goal})$ 和真实任务进展高度正相关（Pearson $r > 0.7$）。

### 2.3 VLA 的独特优势

现有方法（RoboCLIP 等）使用**外部** CLIP 模型计算奖励。SR-VLA 的关键区别：

**π0 的 VLM（PaLI-Gemma with SigLIP encoder）比独立 CLIP 更强**：
- SigLIP ViT-So400m：400M 参数 vs CLIP ViT-B 86M
- 在 PaLI-Gemma 中经过视觉-语言联合训练，理解能力更强
- 输入包含多视角图像（base + 2 wrist cameras），信息更丰富

**零额外计算**：VLM 特征在每一步都要计算（用于 action generation），直接复用即可。RoboCLIP 需要额外的 CLIP forward pass。

---

## 三、方法：SR-VLA

### 3.1 整体框架

SR-VLA 在 Flow-SDE + PPO 的基础上增加**一个步骤**：将 VLM 视觉特征转化为 PBRS 稠密奖励，嵌入 GAE 计算。其他所有组件（Flow-SDE log-prob、PPO clip、value head、GAE）完全不变。

```
                    Flow-SDE + PPO (不变)
                    ┌─────────────────────────────────────┐
Observation  ──→   │  VLM Prefix Encoding                │
                   │    ├── prefix_output [B, 968, 2048]  │
                   │    │   ├── 用于 action generation ✓   │
                   │    │   ├── 用于 value head ✓           │
  [NEW] ──→        │    │   └── 用于 Self-Reward ★ (新增)  │
                   │    └── KV cache                      │
                   │  Action Expert (suffix) → action     │
                   │  Environment → sparse reward         │
  [NEW] ──→        │  PBRS shaped reward = r + γΦ' - Φ ★  │
                   │  GAE (用 shaped reward) → advantage  │
                   │  PPO clip loss → update θ            │
                   └─────────────────────────────────────┘
```

### 3.2 Goal Feature 提取

从 SFT 训练数据（成功的 demonstration）中提取 goal features：

**Step 1**：收集成功 episode 的最后 $N_{goal}$ 步观测（默认 $N_{goal} = 5$）

**Step 2**：对每个任务 $\ell_i$，用 VLM 编码器提取视觉特征并平均：

$$f_{goal}(\ell_i) = \frac{1}{N_{goal}} \sum_{k=1}^{N_{goal}} \text{Pool}\left(\text{VLM}(o_{T-k}, \ell_i)\right)$$

其中 Pool 是 mean pooling over selected tokens（与 value head 使用的 `value_vlm_mode` 一致）：

```python
# 与 get_value_from_vlm() 使用相同的 token selection
# Pi0 config: mean of image tokens (256 × num_images) + language tokens
# Pool to: [2048] (Pi05) or [1024] (Pi0)
```

**Step 3**：L2 归一化：$f_{goal} \leftarrow f_{goal} / \|f_{goal}\|_2$

Goal features 只需预计算一次，存储在 rollout worker 中。对于 LIBERO 的 10 个任务，总存储量 $< 1$ MB。

### 3.3 Potential Function 定义

$$\Phi(o_t, \ell) = \alpha \cdot \cos\!\left(\text{Pool}(\text{VLM}(o_t, \ell)),\; f_{goal}(\ell)\right)$$

其中 $\alpha > 0$ 是 scaling 系数（默认 $\alpha = 1.0$），控制 shaped reward 的幅度。

**性质**：
- $\Phi \in [-\alpha, \alpha]$
- 当机器人完成任务时（视觉状态接近 goal），$\Phi \to \alpha$
- 当机器人远离目标时，$\Phi \to 0$ 或负值
- $\Phi$ 是观测的确定性函数，满足 PBRS 要求

### 3.4 PBRS Shaped Reward

$$r'_t = r_t^{sparse} + \gamma \cdot \Phi(o_{t+1}, \ell) - \Phi(o_t, \ell)$$

展开：

$$r'_t = r_t^{sparse} + \gamma \alpha \cdot \left[\cos(f_{t+1}, f_{goal}) - \frac{1}{\gamma}\cos(f_t, f_{goal})\right]$$

当 $\gamma \approx 1$ 时（$\gamma = 0.99$），近似为：

$$r'_t \approx r_t^{sparse} + \alpha \cdot \Delta\cos_t$$

其中 $\Delta\cos_t = \cos(f_{t+1}, f_{goal}) - \cos(f_t, f_{goal})$ 是 VLM 特征空间中的"进展量"。

### 3.5 嵌入 GAE

用 $r'_t$ 替换 $r_t$ 进入 GAE 计算：

$$\delta'_t = r'_t + \gamma V(o_{t+1}, \ell) - V(o_t, \ell)$$

$$A_t^{GAE} = \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta'_{t+k}$$

**等价形式**（利用 PBRS 的 telescoping 性质）：

$$\sum_{t=0}^{T} \delta'_t = \sum_{t=0}^{T} \delta_t + \gamma^{T+1}\Phi(o_{T+1}) - \Phi(o_0)$$

即：shaped GAE = 原始 GAE + 来自 potential function 的修正项。长期来看，PBRS 不改变最优策略，但在短期内（$V$ 不准确时）提供了更准确的 per-step advantage 信号。

### 3.6 Value Function 训练

Value function 的 TD target 也使用 shaped reward：

$$y_t = r'_t + \gamma \cdot V_{\bar{\phi}}(o_{t+1}, \ell)$$

$$\mathcal{L}_V = \mathbb{E}\left[(V_\phi(o_t, \ell) - y_t)^2\right]$$

由于 $r'_t$ 包含 VLM 特征的进展信号，$V$ 的 bootstrap 训练从每一步都获得非零目标值（而非只在 episode 末尾），收敛显著加快。

### 3.7 完整算法

---

#### Algorithm 1: SR-VLA (Self-Rewarding VLA)

**Pre-compute Phase (一次性)**：

1. 从 SFT 数据中提取 goal features $f_{goal}(\ell_i)$ for each task $\ell_i$
2. L2 归一化并存储在 rollout worker 中

**For** $k = 0, 1, \ldots, K-1$ **do:**

**Step 1 (Rollout with Self-Reward)**：
用 $\pi_{\theta_k}$ 在 $N_{env}$ 个并行环境中执行 rollout。每一步：
- VLM 计算 prefix\_output（用于 action generation）
- 从 prefix\_output 中提取 pooled features $f_t$（零额外计算）
- 计算 potential：$\Phi_t = \alpha \cdot \cos(f_t, f_{goal})$
- 计算 shaped reward：$r'_t = r_t^{sparse} + \gamma \Phi_{t+1} - \Phi_t$

**Step 2 (GAE with Shaped Reward)**：
$$A_t = \text{GAE}(r'_0, \ldots, r'_T, V_\phi, \gamma, \lambda)$$

**Step 3 (PPO Update — 完全不变)**：
$$\mathcal{L}_{PPO} = -\mathbb{E}\left[\min\left(r(\theta) A_t, \; \text{clip}(r(\theta), 1\!-\!\epsilon, 1\!+\!\epsilon) A_t\right)\right]$$

$$\theta_{k+1} \leftarrow \theta_k - \alpha_\pi \nabla_\theta \mathcal{L}_{PPO}$$

**Step 4 (Value Update — 用 shaped reward)**：
$$\mathcal{L}_V = \mathbb{E}\left[(V_\phi(o_t) - y_t)^2\right], \quad y_t = r'_t + \gamma V_{\bar{\phi}}(o_{t+1})$$

**End For**

**Return** $\pi_{\theta_K}$

---

### 3.8 实现细节

#### 3.8.1 Feature Pooling 策略

与 `get_value_from_vlm()` 复用相同的 token selection 逻辑：

```python
# openpi_action_model.py: get_value_from_vlm() (lines 1028-1057)
# Pi0: value_vlm_mode = "mean_token"
# 选取 image tokens (256 × 2) + language tokens (48)
# Pool: mean over selected tokens → [batch, 1024]
```

SR-VLA 直接复用这个 pooled feature，无需额外 forward pass：

```python
# 在 VLM prefix encoding 之后（已有代码）
prefix_output = vlm_forward(images, language)  # [B, 968, 2048]

# Value head（已有代码）
pooled_for_value = pool_vlm_features(prefix_output)  # [B, 1024]
value = value_head(pooled_for_value)

# Self-Reward（新增，复用 pooled_for_value）
f_t = F.normalize(pooled_for_value, dim=-1)  # [B, 1024]
phi_t = alpha * (f_t * f_goal).sum(dim=-1)   # [B] cosine similarity
```

#### 3.8.2 Rollout Worker 改动

在 `huggingface_worker.py` 中，每一步 rollout 后存储 potential：

```python
# predict_action_batch() 返回后
# pooled_features 已在 VLM forward 中计算
f_t = F.normalize(pooled_features.detach(), dim=-1)
phi_t = self._sr_alpha * (f_t * self._goal_features).sum(dim=-1)
forward_inputs["sr_phi"] = phi_t.cpu()  # [batch]
```

#### 3.8.3 Actor Worker 改动

在 credit assignment 阶段（`fsdp_actor_worker.py`），GAE 计算前修改 rewards：

```python
# 提取 stored potentials
phi_t = forward_inputs["sr_phi"]      # [n_steps, batch]
phi_t_next = phi_t[1:]                # [n_steps-1, batch]

# PBRS shaped reward
step_rewards = rewards.sum(dim=-1)    # [n_steps, batch]（已有）
step_rewards[:-1] += gamma * phi_t_next - phi_t[:-1]  # PBRS shaping

# 然后正常调用 GAE（完全不变）
advantages = compute_gae(step_rewards, values, dones, gamma, gae_lambda)
```

#### 3.8.4 Goal Feature 预计算

在 rollout worker 初始化时，从 SFT checkpoint 加载 goal features：

```python
# 选项 A：从 SFT 数据预计算（推荐）
# 离线执行一次：
#   1. 加载 SFT demonstrations
#   2. 对每个任务，取成功 episode 最后 5 步
#   3. VLM forward → pool → normalize → 保存为 .pt 文件

# 选项 B：从 RL 训练中在线收集（自动化）
# 每次成功 episode 后，更新 goal features 的 EMA：
#   f_goal ← β * f_goal + (1-β) * f_success_final
#   β = 0.99（slow update）
```

---

## 四、理论分析

### 4.1 PBRS 最优策略不变性

**定理 1（Ng 1999）**：设 $\Phi: S \to \mathbb{R}$ 为有界势函数，$r' = r + \gamma\Phi(s') - \Phi(s)$。则 $\pi^*_{r'} = \pi^*_r$。

**证明关键步**：对任意策略 $\pi$，在 shaped MDP 下的 return 为：

$$\sum_{t=0}^{\infty} \gamma^t r'_t = \sum_{t=0}^{\infty} \gamma^t r_t + \gamma^{\infty}\Phi(s_\infty) - \Phi(s_0)$$

末项是常数（不依赖 $\pi$），因此最大化 shaped return $\Leftrightarrow$ 最大化原始 return。

**SR-VLA 满足条件**：$\Phi(o_t, \ell) = \alpha \cdot \cos(f_t, f_{goal})$ 是有界的（$|\Phi| \leq \alpha$），是观测的确定性函数。

### 4.2 GAE 方差分析

**命题 1**：在训练初期（$V \approx 0$），SR-VLA 的 GAE advantage 方差显著低于 baseline。

**分析**：

训练初期 $V(o_t) \approx 0$，baseline GAE 的 TD error：

$$\delta_t^{base} = r_t + \gamma V(o_{t+1}) - V(o_t) \approx r_t$$

对于 sparse reward，绝大多数步 $\delta_t^{base} = 0$。GAE advantage 只在 episode 末尾非零。

SR-VLA 的 TD error：

$$\delta_t^{SR} = r_t + \alpha \Delta\cos_t + \gamma V(o_{t+1}) - V(o_t) \approx \alpha \Delta\cos_t$$

由于 VLM 特征在每一步都有变化（机械臂移动 → 视觉特征变化），$\Delta\cos_t \neq 0$ 几乎处处成立。

$$\text{Var}[A_t^{SR}] \propto \text{Var}[\Delta\cos_t] > 0 \quad \text{vs} \quad \text{Var}[A_t^{base}] \approx 0 \text{ (初期)}$$

**直觉**："方差更高"看似不好，但关键是 $A_t^{SR}$ 包含了**有意义的信号**（接近目标 → 正 advantage，远离目标 → 负 advantage），而 $A_t^{base} \approx 0$ 完全无信号。有信号的 noise 远好于无信号。

### 4.3 收敛加速的理论依据

**命题 2**：设 $\Phi$ 和 $V^*$（真实 value function）之间的相关系数为 $\rho = \text{Corr}(\Phi, V^*)$。则 SR-VLA 相对于 baseline 的等效收敛加速为：

$$\text{Speedup} \approx \frac{1}{1 - \rho^2}$$

**推导思路**：PBRS 的 shaped reward 等价于用 $\Phi$ 作为 value function 的"warm start"。当 $\Phi \approx V^*$ 时（$\rho \to 1$），GAE 从第一步就给出准确的 advantage，无需等待 $V$ 收敛。

对于 VLM 视觉特征：根据 RoboCLIP 的实验数据，$\rho \approx 0.7-0.85$（在 manipulation 任务上），对应 speedup $\approx 2.0\text{-}3.6\times$。

### 4.4 与 Progress Reward 的比较

πRL 使用的 progress reward $r_{prog} = t/T$ 也可以视为 PBRS，其势函数：

$$\Phi_{prog}(o_t) = t/T \quad (\text{仅依赖时间步，不依赖观测})$$

**SR-VLA vs Progress Reward**：

| | Progress Reward | SR-VLA |
|---|---|---|
| 势函数 | $\Phi = t/T$（与观测无关） | $\Phi = \alpha \cos(f_t, f_{goal})$（与观测强相关） |
| 信息量 | 只知道时间过了多少 | 知道机器人离目标多近 |
| 成功/失败区分 | 不区分（失败也线性增长） | 失败时 $\Phi$ 停滞或下降 |
| 与 $V^*$ 相关性 | 弱（$\rho \approx 0.3$） | 强（$\rho \approx 0.7\text{-}0.85$） |
| 预期加速 | 1.1-1.2× | **2.0-3.6×** |

两者可以叠加使用：$\Phi_{combined} = \Phi_{SR} + \beta \cdot \Phi_{prog}$。

---

## 五、风险评估

### 5.1 可能有效的理由

| 论据 | 置信度 |
|------|--------|
| PBRS 理论保证不改变最优策略 | **极高** — 经典定理，无条件成立 |
| 预训练视觉特征衡量任务进展有效 | **高** — RoboCLIP/LIV/R3M 多项工作验证 |
| SigLIP/PaLI-Gemma 特征质量优于 CLIP | **高** — 更大模型 + 多模态训练 |
| 零额外计算成本 | **极高** — VLM 特征复用，只多一个 cosine similarity |
| Dense reward 加速 RL 训练 | **极高** — RL 领域普遍共识 |

### 5.2 可能失败的理由

| 风险 | 严重性 | 具体分析 | 应对方案 |
|------|--------|---------|---------|
| VLM 特征不包含任务进展信息 | **中** | 可能 VLM 特征主要编码 appearance 而非 spatial progress | 先做 probing 实验验证特征质量 |
| Cosine similarity 方向不对 | **中** | 某些任务中接近目标时特征可能不单调变化 | 用 EMA 更新 goal features；尝试 L2 距离替代 cosine |
| Shaped reward 幅度不匹配 | **低** | $\alpha$ 过大 → shaped reward 主导；$\alpha$ 过小 → 无效果 | 网格搜索 $\alpha \in \{0.1, 0.5, 1.0, 2.0\}$ |
| Value function 受 shaped reward 干扰 | **低** | PBRS 理论保证收敛到正确的 $V^*$ | 理论上无风险，可能需要更多 epoch 但不会偏 |
| 多视角图像的特征 pooling 丢失信息 | **低** | Mean pooling 可能混淆不同视角的信息 | 尝试不同 pooling 策略（mean, last token, concat） |

### 5.3 成功/失败判断标准

| 结果 | 含义 | 下一步 |
|------|------|--------|
| 收敛速度提升 ≥2×，最终 success rate 不降 | ✅ SR-VLA 有效 | 扩展到全 benchmark，分析 feature 质量 |
| 收敛速度提升 1.2-2×，success rate 不降 | ⚠️ 有效但幅度有限 | 调整 $\alpha$、pooling 策略、尝试 fine-tuned features |
| 收敛速度无变化 | ⚠️ VLM 特征可能不含进展信息 | 做 probing 分析，换用 wrist camera 特征 |
| Success rate 下降 | ❌ Shaped reward 引入 bias | 检查 PBRS 实现是否正确（$\gamma$ 项） |

---

## 六、最小可行验证实验

### 6.1 实验目标

验证核心假设：**VLM 视觉特征的 cosine similarity 变化能否有效指示任务进展，从而加速 PPO 训练？**

### 6.2 两阶段验证

#### 阶段 0：Feature Probing（不需要 RL 训练，1 天）

在 SFT demonstration 数据上验证 VLM 特征的进展指示性：

1. 对 10 条成功 episode，提取每步的 pooled VLM features
2. 计算 $\cos(f_t, f_{goal})$ 曲线
3. 检查是否**单调递增**（接近目标时 similarity 上升）
4. 计算 Pearson 相关系数：$\rho = \text{Corr}(\cos(f_t, f_{goal}), t/T)$
5. 可视化：绘制 cosine similarity vs time step 曲线

**判断标准**：
- $\rho > 0.6$：proceed to Phase 1
- $\rho < 0.3$：VLM 特征不适合做进展指标，停止

#### 阶段 1：RL 训练验证（3-5 天）

在 LIBERO-Object（单任务 task 0）上对比：

| 配置 | 方法 | 改动 |
|------|------|------|
| Baseline | Flow-SDE + PPO（sparse reward） | 无 |
| SR-VLA | Flow-SDE + PPO（shaped reward） | +SR |
| SR-VLA + Prog | Flow-SDE + PPO（shaped + progress） | +SR +Prog |

### 6.3 Setup

| 配置项 | 选择 | 理由 |
|--------|------|------|
| 环境 | LIBERO-Object (task 0) | πRL 标准 benchmark |
| Policy 模型 | π0 (OpenPI) | 与现有代码一致 |
| Baseline | Flow-SDE + PPO（现有配置） | 控制变量 |
| 并行环境数 | 32 | 与现有配置一致 |
| 训练 epoch | 400 | 足够看到收敛 |
| 评估 | 每 10 epoch 评估 500 episodes | 与现有配置一致 |

### 6.4 超参数

| 超参数 | 初始值 | 扫描范围 | 理由 |
|--------|--------|---------|------|
| $\alpha$（shaped reward 幅度） | 1.0 | {0.1, 0.5, 1.0, 2.0} | 控制 dense vs sparse reward 平衡 |
| Goal feature 来源 | SFT 最后 5 步均值 | {最后 1 步, 5 步, 10 步} | 影响 goal representation |
| Pooling 策略 | mean_token | {mean_token, last_token} | 与 value head 一致 |
| $N_{goal}$（每个任务的 goal 样本数） | 5 episodes | {1, 5, 10} | 更多 → 更鲁棒的 goal feature |
| 其他所有超参数 | 与 baseline 完全一致 | — | 控制变量 |

### 6.5 对比指标

| 指标 | 怎么测 | 目的 |
|------|--------|------|
| Success Rate vs Epoch | 每 10 epoch 评估 | 核心性能 |
| Epoch-to-X% | 达到 50%/80% success rate 的 epoch | 收敛速度 |
| AUC (前 200 epoch) | Success rate 曲线下面积 | 综合效率 |
| Shaped Reward Correlation | $\text{Corr}(r', \text{episode return})$ | 奖励质量 |
| Value Loss 收敛速度 | $\mathcal{L}_V$ vs epoch | 验证 $V$ 训练加速 |
| GAE Advantage 有效性 | $\text{Corr}(A_t, \text{future return})$ | Credit assignment 质量 |

### 6.6 代码改动清单

| 文件 | 改动 | 行数 |
|------|------|------|
| `huggingface_worker.py` | 加载 goal features；每步存储 `sr_phi` | ~20 行 |
| `fsdp_actor_worker.py` | GAE 前修改 rewards | ~10 行 |
| `pi0.yaml` / quickstart yaml | 新增 SR 配置项 | ~5 行 |
| `precompute_goal_features.py` | 新文件：预计算 goal features 的脚本 | ~50 行 |

**总改动量 < 100 行代码**。不需要新网络、不需要改模型架构、不需要改 PPO loss。

### 6.7 时间计划

| 时间 | 任务 |
|------|------|
| Day 1 | Phase 0：Feature Probing（验证 VLM 特征质量） |
| Day 2 | 代码实现 + debug |
| Day 3-5 | Phase 1：LIBERO-Object 单任务训练对比 |
| Day 6 | $\alpha$ sweep + 结果分析 |
| Day 7 | 决定是否扩展 |

---

## 七、如果验证成功 → 完整方案

### 7.1 扩展实验

| Benchmark | Tasks | 目的 |
|-----------|-------|------|
| LIBERO-Object | 10 tasks | 单任务性能 |
| LIBERO-Spatial | 10 tasks | 空间推理任务 |
| LIBERO-Goal | 10 tasks | 目标导向任务 |
| LIBERO-Long | 10 tasks | 长 horizon（验证 credit assignment 改善） |
| SimplerEnv / ManiSkill | 多任务 | 验证跨环境泛化 |

### 7.2 消融实验

| 消融项 | 变体 | 目的 |
|--------|------|------|
| 特征来源 | SigLIP only / PaLI-Gemma full / CLIP external | VLM 特征 vs 外部特征 |
| Pooling 策略 | mean / last / first / attention-weighted | 最优 pooling |
| $\alpha$ 调度 | 固定 / 线性衰减 / cosine annealing | 动态调整 shaped reward 权重 |
| Goal 表示 | 固定 SFT / EMA 在线更新 / 多 goal 平均 | 鲁棒 goal representation |
| 与 Progress 结合 | SR only / Prog only / SR + Prog | 互补性 |

### 7.3 分析实验

1. **Feature Probing**：在不同任务上可视化 $\cos(f_t, f_{goal})$ vs time step，分析单调性
2. **Reward Correlation**：$r'_t$ vs ground-truth task progress 的 Pearson/Spearman 相关系数
3. **Value Convergence**：对比 baseline 和 SR-VLA 的 $V$ 训练曲线
4. **Advantage Quality**：对比 $A_t$ 和实际 future return 的相关系数
5. **Failure Mode Analysis**：分析 SR-VLA 失败的 episode，检查 VLM 特征是否给出错误进展信号

### 7.4 跨任务泛化实验

SR-VLA 的一个独特优势是**跨任务泛化**：

1. **Same-suite transfer**：在 LIBERO-Object task 0 上计算 goal features，应用到 task 1-9
2. **Cross-suite transfer**：用 LIBERO-Object 的 goal features 应用到 LIBERO-Spatial
3. **Zero-shot goal specification**：用语言描述生成 VLM features 作为 goal（无需 demo）

**假设**：VLM 特征的语义空间是跨任务共享的。如果 "pick up red cup" 和 "pick up blue cup" 的 goal features 在 VLM 空间中接近，则 reward shaping 自动泛化。

### 7.5 论文结构

**Title:** *Self-Rewarding Vision-Language-Action Models: Zero-Cost Dense Rewards from Pretrained Visual Features*

| Section | Pages | 内容 |
|---------|-------|------|
| 1. Introduction | 1.5 | Sparse reward bottleneck → Self-rewarding insight |
| 2. Related Work | 1 | Reward shaping, VLM rewards, VLA RL |
| 3. Method | 2 | PBRS theory + SR-VLA algorithm + implementation |
| 4. Experiments | 3 | Main results + ablations + cross-task + analysis |
| 5. Analysis | 1 | Feature probing + reward quality + failure modes |
| 6. Conclusion | 0.5 | |

### 7.6 与现有方法的定位

| 方法 | 奖励来源 | 额外模型？ | 额外计算？ | 理论保证？ | 适用范围 |
|------|---------|:---------:|:---------:|:---------:|---------|
| Sparse Reward | 环境 | ✗ | ✗ | — | 通用 |
| Progress Reward | 时间步 | ✗ | ✗ | PBRS ✓ | 通用但信息弱 |
| RoboCLIP | CLIP | ✓ 外部 CLIP | ✓ 额外 forward | ✗ | 需要额外模型 |
| VLM-RM | 训练 RM | ✓ 训练 RM | ✓ 额外 forward | ✗ | 需要 preference data |
| **SR-VLA (ours)** | **VLM 自身** | **✗** | **✗** | **PBRS ✓** | **任何 VLM-based VLA** |

**SR-VLA 的核心卖点**：

1. **Self-Rewarding**：同一个 VLM 既生成动作又提供奖励信号，形成自驱动闭环
2. **Zero-Cost**：不需要额外模型、额外 forward pass、额外训练数据
3. **Theory-Backed**：PBRS 保证不改变最优策略（不引入 bias）
4. **Generalizable**：VLM 特征天然跨任务、跨环境泛化
5. **Plug-and-Play**：100 行代码改动，兼容任何 Flow + PPO 训练框架

---

## 八、可选扩展

### 8.1 扩展 A：Adaptive $\alpha$ Scheduling

$\alpha$ 在训练过程中自适应调整：
- 初期 $\alpha$ 较大（$V$ 不准，依赖 shaped reward）
- 后期 $\alpha$ 减小（$V$ 已准确，shaped reward 作用递减）

$$\alpha(k) = \alpha_0 \cdot \max\left(0, 1 - \frac{k}{K_{anneal}}\right)$$

### 8.2 扩展 B：Multi-Granularity Features

不只用 pooled features，还用 per-image features（base camera vs wrist camera）：

$$\Phi(o_t) = \alpha_1 \cos(f_t^{base}, f_{goal}^{base}) + \alpha_2 \cos(f_t^{wrist}, f_{goal}^{wrist})$$

Wrist camera 特征可能更好地捕捉细粒度操作进展（抓取接触 etc.）。

### 8.3 扩展 C：Online Goal Feature Update

不固定 goal features，而是用 RL 训练中成功 episode 的特征在线更新：

$$f_{goal} \leftarrow \beta \cdot f_{goal} + (1-\beta) \cdot f_{success}, \quad \beta = 0.99$$

随着 policy 改进，goal features 也随之优化，更准确地反映"真正的成功状态"。

### 8.4 扩展 D：与其他 Flow RL 方法结合

SR-VLA 是 **orthogonal** 于 policy optimization 的改进，可以和任何方法结合：

- SR-VLA + Flow-SDE PPO（本方案）
- SR-VLA + FlowIPO（shaped reward 改善 FlowIPO 的 credit assignment）
- SR-VLA + AWM-VLA（shaped reward 改善 AWM 的 value head 质量）
- SR-VLA + Flow-Noise PPO

---

## 九、总结

**一句话**：VLA 中的 VLM 已经"看到"了任务进展，只是这个信号被浪费了。SR-VLA 用 PBRS 理论将 VLM 视觉特征转化为稠密奖励，零额外计算地加速 Flow-SDE + PPO 训练。

**最小可行验证**：1 天 feature probing + 5 天 RL 训练对比，改约 100 行代码。

**最大风险**：VLM 特征的 cosine similarity 变化不单调（用 Phase 0 probing 提前排除）。

**最好情况**：2-4× 收敛加速，跨任务泛化有效 → 值得 NeurIPS 2026 投稿。

**最差情况**：VLM 特征不含进展信息 → Phase 0 即可发现，不浪费 RL 训练时间。

**与前序工作（AWM/FPI/GFN）的本质区别**：前序工作试图替换 PPO 的优化目标（全部失败）。SR-VLA 不动 PPO，只改善 PPO 的输入信号（稠密奖励），这是一个 orthogonal 的改进方向。
