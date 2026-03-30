# Hinge-NFT：对称Margin Loss解锁Contrastive Flow RL的Temporal Credit

> **核心主张**：π-StepNFT在long-horizon任务上的性能gap不是因为mirror construction或sample-space energy有问题，而是因为softplus loss的**不对称性**阻止了任何形式的temporal credit。将softplus替换为对称的hinge loss，contrastive flow RL就能安全接入per-step信用分配——无需value network，无需跨轨迹对比。

> **一句话**："Softplus is the bottleneck, not the credit."

---

## 一、Problem Statement

### 1.1 π-StepNFT的唯一短板

π-StepNFT（ICML 2026）解决了flow-based VLA RL的梯度消失问题（梯度 $\propto (v_\theta - v_{\text{old}})$，不随预训练衰减），在short/medium任务上与PPO持平，OOD泛化上反超PPO。

但在**long-horizon任务**上有~4% gap（Table 1, π-StepNFT）：

| 任务类型 | π-StepNFT | PPO | Gap |
|----------|-----------|-----|-----|
| Short | 98.2% | 98.4% | -0.2% |
| Long | 86.0% | 90.2% | **-4.2%** |
| OOD | 50.4% | 39.3% | **+11.1%** |

### 1.2 Gap的根因：梯度稀释

Terminal binary advantage $y = \pm 1$ 对所有env step施加**相同权重**的梯度。在300步的long任务中：

- 关键决策点（抓取、放置）：~30步
- Routine步骤（接近、等待）：~270步
- 关键步的梯度被稀释 $\frac{30}{300} = \frac{1}{10}$

PPO通过GAE + value function给关键步更大的advantage，从而集中梯度。NFT没有这个机制。

### 1.3 为什么不能直接给NFT加temporal credit？

我们在FlowIPO框架中实验了两种方案，**全部失败**：

| 方案 | 思路 | 结果 | 根因 |
|------|------|------|------|
| FEA v1 | Ridge regression → TD advantage → continuous $y$ | 训练崩溃 ❌ | TD≈0 → 归一化后随机 $\pm 1$ 标签 |
| FEA v2 | Terminal sign × |TD| importance | 慢于terminal binary ⚠️ | 失败trajectory $V \approx 0$ → importance ≈ 0.2 → 信号削弱到1/5 |

**根本原因不是temporal credit方案本身，而是softplus loss**：

$$\text{softplus}(x) \approx \begin{cases} e^x \to 0 & x \ll 0 \quad \text{（正确label → 指数饱和）} \\ x & x \gg 0 \quad \text{（错误label → 线性增长无上界）} \end{cases}$$

这个不对称性意味着：
1. **连续 $|y|$ 不安全**：$|y|$ 小的step，如果label有微小噪声，softplus的线性端造成的damage远超饱和端的benefit
2. **标签噪声灾难**：~50%随机错误标签时，错误端的loss $\approx |\Delta E|$（线性），正确端的loss $\approx 0$（饱和）→ 净效果是**反方向训练**
3. **信号不均衡**：FEA v2的importance对失败trajectory系统性偏低 → push-pull不均衡 → 学习减慢

**结论**：softplus是NFT无法做temporal credit的**唯一瓶颈**。

### 1.4 核心洞察

> **将softplus替换为对称的hinge loss**，所有temporal credit方案都变安全了——因为hinge对正确和错误标签的惩罚是对称的，random label noise的期望梯度为零，不会反转训练方向。

---

## 二、Background

### 2.1 π-StepNFT回顾

核心组件（全部保留在Hinge-NFT中）：

1. **SDE链快照**：K步SDE去噪，随机选一步 $k$ 保存 $(x_t^{(k)}, v_{\text{old}}^{(k)}, x_{\text{next}}^{(k)})$
2. **Mirror construction**：$v^+ = v_{\text{old}} + \beta \cdot \Delta v_{\text{clip}}, \quad v^- = v_{\text{old}} - \beta \cdot \Delta v_{\text{clip}}$
3. **Sample-space Mahalanobis energy**：$E^{\pm} = \sum_{\text{dim}} \frac{(x_{\text{next}} - \mu^{\pm})^2}{\sigma^2}$
4. **KL trust region**：$\|v_\theta - v_{\text{old}}\|^2$

**Hinge-NFT只改第5项**（loss函数），其余全部不变。

### 2.2 VLM Embedding Change Rate（方案B）

VLA模型的VLM prefix encoder会为每个observation产出一个embedding $e_t$。这个embedding天然编码了**场景的视觉语义信息**。帧间变化率：

$$w_t = \|e_{t+1} - e_t\|_2$$

天然反映了场景的**关键程度**：

| 阶段 | $\|e_{t+1} - e_t\|$ | 权重 | 说明 |
|------|---------------------|------|------|
| 手臂接近物体 | 小 | 低 | 视觉缓慢变化，routine |
| **抓取物体** | **大** | **高** | 物体位置/姿态突变，关键决策点 |
| 搬运中 | 中 | 中 | 稳定移动 |
| **放置物体** | **大** | **高** | 又一次场景突变，关键决策点 |

**关键优势**：
1. **零额外成本**：VLM embedding在rollout的prefix encoding后顺手收集（`mean_pool(prefix_output).detach()`），不需要EMA、不需要额外forward pass
2. **成功/失败均衡**：embedding change rate与episode outcome无关——无论成功还是失败，抓取瞬间的embedding变化都很大。不存在FEA v2中失败trajectory信号被系统性削弱的问题
3. **不需要回归**：只用L2距离的相对大小，不需要拟合任何模型，鲁棒性高

### 2.3 为什么之前不能用连续advantage

在NFT的softplus loss中，即使方案B的importance信号完全正确（sign来自terminal binary，magnitude来自embedding change），softplus的不对称性仍然会扭曲连续 $|y|$ 值的梯度贡献。

Hinge loss修复了这个问题——对称的梯度意味着连续 $|y|$ 值的贡献与 $|y|$ 成正比，不会被softplus的饱和端压制。

---

## 三、Method

### 3.1 系统架构

Hinge-NFT = **NFT的数据收集 + VLM embedding步级信用分配 + Hinge loss**

```
┌─ Stage 1: Rollout（数据收集）─────────────────────────────────────────────┐
│  openpi_action_model.py :: sample_actions(collect_flow_snap=True)         │
│                                                                           │
│  (a) 独立SDE去噪循环 → SDE快照 (nft_xt, nft_v, nft_xnext, ...)         │
│  (b) vlm_embedding = mean_pool(prefix_output).detach()  ← 零额外成本    │
│  (c) 最终动作 x_0（环境执行用）                                          │
│  (d) episode_reward (env 返回的 0/1 binary reward)                       │
│                                                                           │
│  ★ 无需EMA参考模型，无需额外forward pass                                 │
└───────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─ Stage 2: Credit Assignment（标签生成）────────────────────────────────────┐
│  fsdp_actor_worker.py :: _compute_flow_hinge_nft_credit_assignment()     │
│                                                                           │
│  embedding_change (默认):                                                │
│    w_t = ||e_{t+1} - e_t||₂  (VLM embedding 帧间 L2 变化率)             │
│    w_norm = clamp(w / mean(w), w_min, w_max)                             │
│    A_t = sign(R) × w_norm × c_adv                                       │
│    → 连续 y ∈ [-1, 1]，hinge loss保证安全                                │
│    → 场景剧变帧(抓取/放置) → 高权重，routine帧 → 低权重                 │
│                                                                           │
│  terminal_binary (消融实验):                                             │
│    A_t = ±c_adv  (与NFT相同，y = ±1)                                    │
│    → 测试 hinge vs softplus 的纯loss效果                                 │
│                                                                           │
│  输出: advantages [T, B]                                                 │
└───────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─ Stage 3: Training（Hinge loss + 反向传播）───────────────────────────────┐
│  losses.py :: compute_flow_hinge_nft_loss()                              │
│                                                                           │
│  (a) 镜像构造: v⁺ = v_old + β·Δv_clip,  v⁻ = v_old - β·Δv_clip        │
│  (b) Sample-space Mahalanobis能量: E⁺, E⁻                              │
│  (c) Label: y = A_t / c_adv ∈ [-1, 1]                                   │
│  (d) Hinge loss: ℓ = max(0, m + ½ y·(E⁺-E⁻))                          │
│  (e) Trust region: β_kl · ||Δv||²                                        │
│                                                                           │
│  与NFT的唯一差别: (d) 从 softplus → hinge                                │
└───────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Hinge Loss（核心改动）

**NFT原版**（softplus）：

$$\ell_t^{\text{sp}}(\theta) = \text{softplus}\!\left(\tfrac{1}{2}\, y \cdot \Delta E\right), \quad \Delta E = E_{\theta,t}^+ - E_{\theta,t}^-$$

**Hinge-NFT**：

$$\ell_t^{\text{hinge}}(\theta) = \max\!\left(0,\; m + \tfrac{1}{2}\, y \cdot \Delta E\right)$$

其中 $m > 0$ 是margin超参数。

**语义**：
- 正确分类且超过margin（$y \cdot \Delta E < -2m$）：loss = 0，无梯度（同softplus）
- 正确分类但不够确信（$-2m < y \cdot \Delta E < 0$）：loss = $m + \frac{1}{2} y \cdot \Delta E$，梯度推向更确信
- 错误分类（$y \cdot \Delta E > 0$）：loss = $m + \frac{1}{2} y \cdot \Delta E$，梯度纠正方向

**梯度**：

$$\frac{\partial \ell_t^{\text{hinge}}}{\partial \theta} = \begin{cases} \frac{1}{2} y \cdot \frac{\partial \Delta E}{\partial \theta} & \text{if } m + \frac{1}{2} y \cdot \Delta E > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 3.3 为什么Hinge对称而Softplus不对称

**定理（非正式）**：对于标签噪声比例 $p$ 的样本，hinge loss的期望梯度偏差为 $O(p \cdot m)$，而softplus的期望梯度偏差为 $O(p \cdot |\Delta E|)$。当 $|\Delta E| \gg m$ 时（训练中后期），softplus的偏差远大于hinge。

**直觉推导**：

考虑一个样本，真实label为 $y^* = +1$。训练到一定程度后，$\Delta E \approx -D$（$D > 0$，模型已学到正确方向）。

| 情况 | 分配的label $y$ | Softplus loss | Hinge loss ($m < D/2$) |
|------|----------------|---------------|------------------------|
| 正确 ($y = +1$) | $\text{sp}(\frac{1}{2}(-D)) \approx e^{-D/2}$ | $\max(0, m - D/2) = 0$ |
| 错误 ($y = -1$) | $\text{sp}(\frac{1}{2}(+D)) \approx D/2$ | $\max(0, m + D/2) = m + D/2$ |

在比例 $p$ 的标签噪声下：

$$\mathbb{E}[\ell^{\text{sp}}] = (1-p) \cdot e^{-D/2} + p \cdot D/2 \approx p \cdot D/2 \quad \text{（因为 $e^{-D/2} \to 0$）}$$

$$\mathbb{E}[\ell^{\text{hinge}}] = (1-p) \cdot 0 + p \cdot (m + D/2) = p(m + D/2)$$

**loss绝对值相近，但看梯度方向**：

Softplus梯度：

$$\mathbb{E}\left[\frac{\partial \ell^{\text{sp}}}{\partial \theta}\right] = (1-p) \cdot \underbrace{\sigma(-D/2)}_{\to 0} \cdot (+1) \cdot \frac{\partial \Delta E}{\partial \theta} + p \cdot \underbrace{\sigma(D/2)}_{\to 1} \cdot (-1) \cdot \frac{\partial \Delta E}{\partial \theta}$$

$$\approx -p \cdot \frac{\partial \Delta E}{\partial \theta} \quad \text{（净方向 = 错误方向！）}$$

Hinge梯度：

$$\mathbb{E}\left[\frac{\partial \ell^{\text{hinge}}}{\partial \theta}\right] = \underbrace{(1-p) \cdot 0}_{\text{已满足margin}} + p \cdot (-1) \cdot \frac{1}{2} \cdot \frac{\partial \Delta E}{\partial \theta}$$

$$= -\frac{p}{2} \cdot \frac{\partial \Delta E}{\partial \theta} \quad \text{（净方向 = 错误方向）}$$

等等，两者都是错误方向？是的——在已经训练好且margin已满足的区域，任何label noise都会产生反方向偏差。但关键区别在**训练中期**（$D$ 还不大、margin还没满足时）：

当 $m > D/2$（margin未满足，训练早中期）：

Softplus：$(1-p) \cdot \sigma(-D/2) \approx (1-p) \cdot 0.3$ 向正确方向，$p \cdot \sigma(D/2) \approx p \cdot 0.7$ 向错误方向。当 $p > 0.3$ 时净方向为错误。

Hinge：$(1-p) \cdot \frac{1}{2}$ 向正确方向，$p \cdot \frac{1}{2}$ 向错误方向。净方向 $= \frac{1-2p}{2} \cdot \frac{\partial \Delta E}{\partial \theta}$。**只有当 $p > 0.5$ 时才反转**。

| 标签噪声 $p$ | Softplus净方向 | Hinge净方向 |
|-------------|--------------|-----------|
| 0% | 正确 ✅ | 正确 ✅ |
| 10% | 正确 ✅ | 正确 ✅ |
| 30% | **反转 ❌** | 正确 ✅ |
| 50% | **反转 ❌** | 中性 ⚠️ |

**FEA v1产生的noise约为50%（完全随机label）→ softplus反转，hinge中性（不学但不崩）。**

**真正的优势**：对于FEA v2或self-annotation这种sign正确、magnitude有噪声的情况（$p \approx 0$），hinge和softplus表现接近。但hinge的优势在于**可以安全地使用更aggressive的temporal credit**（因为即使偶尔sign错了也不会灾难性崩溃），而softplus必须保守地只用terminal binary。

### 3.4 VLM Embedding Change Rate 信用分配

这是Hinge-NFT的第二个核心组件。使用VLM冻结embedding的**帧间变化率**作为per-step importance，label方向始终来自terminal binary。

#### Step 1：收集embedding（rollout时零成本）

VLM prefix encoder对每个observation $o_t$ 产出embedding，在rollout的`sample_actions()`中顺手收集：

$$e_t = \text{mean\_pool}(\text{prefix\_output}).\text{detach}() \quad \in \mathbb{R}^{B \times d}$$

#### Step 2：计算帧间变化率

$$w_t = \|e_{t+1} - e_t\|_2 \quad \text{（帧间L2距离）}$$

#### Step 3：归一化 + 裁剪

$$\tilde{w}_t = \text{clamp}\!\left(\frac{w_t}{\bar{w}},\; w_{\min},\; w_{\max}\right) \quad \text{（归一化到均值1，裁剪极端值）}$$

#### Step 4：生成advantage

$$A_t = \text{sign}(R - 0.5) \times \tilde{w}_t \times c_{\text{adv}}$$

#### 为什么embedding change在Hinge下有效

1. **Label方向（sign）**：始终来自terminal binary → 永远正确
2. **Label幅度（$|y|$）**：由帧间变化率调节 → 连续值 $\in [w_{\min} \cdot c_{\text{adv}},\; w_{\max} \cdot c_{\text{adv}}]$
3. **Hinge保证**：连续 $|y|$ 的梯度贡献与 $|y|$ 成正比（对称），不会像softplus那样被饱和端压制
4. **成功/失败均衡**：embedding change rate与outcome完全无关——无论成功还是失败，抓取瞬间的embedding变化都一样大

#### 与FEA的关键区别

| | FEA v2 | Embedding Change Rate |
|---|---|---|
| Importance来源 | Ridge regression → TD error ($V$-based) | 帧间embedding L2距离 |
| 失败trajectory | $V \approx 0$ → importance ≈ 0.2（信号削弱） | 正常（与outcome无关） |
| 成功/失败均衡 | 不均衡（失败被系统性压低） | **均衡** |
| 额外网络 | 无 | 无 |
| 额外计算 | Ridge regression（closed-form，快） | **零**（embedding已在rollout时收集） |
| 标签噪声风险 | 中（TD residual有噪声） | **极低**（L2距离直接可观测） |
| EMA参考模型 | 不需要 | **不需要** |

### 3.5 完整Hinge-NFT Loss

$$\ell_t(\theta) = \max\!\left(0,\; m + \tfrac{1}{2}\, y \cdot (E_{\theta,t}^+ - E_{\theta,t}^-)\right)$$

$$\mathcal{L} = \ell_t(\theta) + \beta_{\text{kl}} \|v_\theta - v_{\text{old}}\|^2$$

其中所有其他组件（mirror construction、energy computation、max_drift clipping）与π-StepNFT完全相同。

---

## 四、与现有方法的对比

| 特性 | PPO | π-StepNFT | FlowSAR | **Hinge-NFT** |
|------|-----|-----------|---------|--------------|
| 梯度幅度 | $O(\varepsilon)$（policy gradient） | $O(1)$（mirror） | $O(\varepsilon)$（velocity-space） | **$O(1)$（mirror）** |
| Value Network | ✅ 需要 | ❌ | ❌ | **❌** |
| Temporal Credit | GAE (step-level) | Terminal binary (episode-level) | Self-annotation (step-level) | **Embedding change (step-level)** |
| 信用分配来源 | 学习的Critic | 无 | 模型重建误差 | **VLM embedding帧间变化率** |
| Loss对标签噪声的鲁棒性 | N/A（policy gradient） | 差（softplus不对称） | 差（softplus变体也不对称） | **好（hinge对称）** |
| Long-horizon | ✅ 好 | ⚠️ 有gap | 未知 | **预期好** |
| OOD泛化 | 差（Critic过拟合） | 好 | 好 | **好** |
| EMA/Annotation Pass | 需要（Critic forward） | 不需要 | 需要（2次forward） | **不需要（零额外计算）** |
| 能量空间 | N/A | Sample-space | Velocity-space | **Sample-space** |

**Hinge-NFT的独特位置**：继承NFT的 $O(1)$ 梯度优势（不随预训练衰减），同时通过hinge loss解锁VLM embedding的zero-cost step-level信用分配。

---

## 五、Risk Assessment

### 5.1 Hinge vs Softplus在terminal_binary下的表现

**风险**：当 $y = \pm 1$（无noise）时，softplus和hinge应该表现接近。hinge可能不会更好。

**缓解**：terminal_binary只是baseline实验。真正的价值在self_annotation模式下——这是softplus做不到的。

**验证**：Phase 1实验（见§六）。

### 5.2 Embedding Change Rate的区分度

**风险**：如果所有step的embedding变化率差异很小，权重退化为uniform → 退化为terminal_binary。

**缓解**：
1. 退化为terminal_binary不比NFT差
2. 具身任务有明确的动作阶段（接近→抓取→搬运→放置），VLM embedding应该有明显的帧间变化差异
3. $w_{\min}=0.2, w_{\max}=2.0$ 的clamp范围允许最多10倍的步间权重差异

### 5.3 Margin $m$ 的敏感性

**风险**：$m$ 过大 → 所有样本都违反margin → 接近无margin的MSE行为。$m$ 过小 → 大多数样本满足margin → 梯度太少。

**缓解**：
1. 默认 $m = 1.0$，预期 $|\Delta E|$ 在训练初期约为 $O(1)$
2. 可以用adaptive margin：$m = \alpha \cdot \text{running\_mean}(|\Delta E|)$

### 5.4 计算开销

**优势**：Embedding change credit的额外计算开销为**零**。
- VLM embedding已在rollout的prefix encoding后收集（`mean_pool(prefix_output).detach()`）
- 不需要EMA参考模型
- 不需要额外forward pass
- 只需在actor worker中做一次 $O(T \times B \times d)$ 的L2距离计算

---

## 六、Experimental Plan

### Phase 1：Hinge vs Softplus（纯loss对比）

- LIBERO-Object上对比：
  - `flow_nft` (softplus + terminal_binary) = baseline
  - `flow_hinge_nft` (hinge + terminal_binary) = 新方法
- 预期：两者性能接近（terminal_binary下softplus和hinge差异不大）
- 目的：验证hinge不引入regression

### Phase 2：Embedding Change Temporal Credit

- LIBERO-Long上对比：
  - `flow_hinge_nft` (hinge + terminal_binary) = baseline
  - `flow_hinge_nft` (hinge + embedding_change) = 完整方案（默认）
  - `flow_nft` (softplus + terminal_binary) = 原始NFT
- 预期：embedding_change > terminal_binary on Long tasks
- 目的：验证VLM embedding步级信用分配在hinge下有效

### Phase 3：Full Benchmark

- LIBERO 5 suites: Short, Long, Spatial, Object, Goal
- 对比：Hinge-NFT (embedding_change), NFT (terminal_binary), PPO
- 预期：
  - Short/Medium: Hinge-NFT ≈ NFT ≈ PPO
  - Long: Hinge-NFT > NFT, 接近PPO
  - OOD: Hinge-NFT > PPO（无Critic过拟合）

### 超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `flow_hinge_nft_margin` | 1.0 | Hinge margin $m$ |
| `flow_hinge_nft_beta` | 1.0 | Mirror construction $\beta$ |
| `flow_hinge_nft_kl_beta` | 0.0001 | KL trust region |
| `flow_hinge_nft_max_drift` | 0.5 | Velocity drift clipping |
| `flow_hinge_nft_noise_level` | 0.2 | SDE noise |
| `flow_hinge_nft_adv_clip_max` | 1.0 | Advantage clipping |
| `flow_hinge_nft_adv_type` | embedding_change | `embedding_change` 或 `terminal_binary` |
| `flow_hinge_nft_emb_w_min` | 0.2 | 权重裁剪下界（routine step ≥ 20% 信号）|
| `flow_hinge_nft_emb_w_max` | 2.0 | 权重裁剪上界（关键step ≤ 200% 信号）|

---

## 七、论文故事

**Title**: "Hinge-NFT: Unlocking Temporal Credit for Contrastive Flow RL via Symmetric Margin Loss"

**Contributions**:
1. **诊断**：softplus不对称性是contrastive flow RL无法做temporal credit的根因（理论+实验）
2. **方案**：Hinge-NFT = hinge margin loss + VLM embedding change rate 步级信用分配
3. **理论**：证明hinge在标签噪声下的梯度偏差远小于softplus（noise threshold: softplus 30% vs hinge 50%）
4. **实验**：在LIBERO Long-horizon任务上缩小与PPO的gap，同时保持OOD泛化优势
5. **零额外成本**：embedding credit不需要EMA、不需要额外forward pass、不需要value network

**Narrative**：
- §1 π-StepNFT解决了梯度消失，但在long任务上有gap（terminal binary稀释关键步梯度）
- §2 Gap的根因不是credit assignment方案不够好，而是softplus阻止了所有连续advantage方案
- §3 Hinge loss对称 → 安全使用连续advantage → 解锁temporal credit设计空间
- §4 VLM自身的embedding变化率提供zero-cost per-step importance（场景剧变 = 关键决策点）
- §5 实验验证：Long任务gap缩小，Short/OOD不退化

---

## 八、总结

Hinge-NFT的核心贡献是一个**精确的归因**：

> π-StepNFT在long-horizon任务上的gap，不是因为缺少value network，不是因为没有temporal credit方案，而是因为**softplus loss的不对称性阻止了任何连续advantage的使用**。

解决方案极简：softplus → hinge，一行公式的修改，解锁了整个temporal credit设计空间。VLM embedding change rate是这个设计空间中最自然的选择——零额外网络、零额外forward pass、零EMA模型，场景剧变帧（抓取/放置）天然对应关键决策点，且与episode outcome完全无关（成功/失败信号均衡）。
