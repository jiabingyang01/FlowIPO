# FlowIPO: Reward-Directed Velocity Interpolation for Online Fine-tuning of Flow-based VLAs

> **一句话总结**：我们提出FlowIPO，一个无需似然计算、无需Critic网络的在线RL微调框架。核心机制是根据奖励信号和策略偏移程度，在当前动作的velocity target和参考策略的velocity之间做自适应插值，将RL信号直接注入flow matching训练目标。

---

## 1. 问题背景与动机

### 1.1 Flow-based VLA的现状

Flow-based VLA模型（π0, π0.5）通过flow matching的迭代去噪生成连续动作chunk，已经成为机器人操控领域的主流架构。其动作生成过程为：

$$A^0 \sim \mathcal{N}(0, I) \xrightarrow{v_\theta, K \text{ steps}} A^1 = a_i$$

其中 $v_\theta(A^\tau, \tau, s)$ 是条件速度场（velocity field），$K$通常为4步去噪，$a_i$是一个action chunk（H=5或10个连续动作）。

SFT训练使用标准的Conditional Flow Matching (CFM) loss：

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{\tau, a_0, \epsilon}\left[\|v_\theta(A^\tau, \tau, s) - u\|^2\right]$$

其中 $A^\tau = \tau a_0 + (1-\tau)\epsilon$，$u = a_0 - \epsilon$。

### 1.2 现有RL方法的困境

SFT后的策略受限于专家数据的窄流形，需要RL来扩展能力边界。但flow-based VLA的RL面临核心难题：

| 方法 | 路线 | 代价 |
|------|------|------|
| πRL (Flow-Noise) | 引入可学习噪声网络，构造离散MDP，计算联合似然 | 额外噪声网络，训练后丢弃 |
| πRL (Flow-SDE) | ODE转SDE，构造两层MDP，逐步高斯似然 | MDP链长度×K，训练慢 |
| πRL (PPO) | 上述似然 + Critic网络 + GAE | Critic过拟合多模态特征，OOD差 |
| GRPO | 同状态采N条轨迹，组内归一化 | 需要似然比，N倍rollout开销 |

**所有现有方法都在试图把flow模型塞进policy gradient框架**——要么硬算似然，要么绕道估计。

### 1.3 核心洞察

> **Flow-based VLA的RL不需要走policy gradient路线。**
> 
> 我们可以直接在velocity target上注入RL信号——通过修改flow matching的训练目标来实现策略改进，完全绕过似然估计问题。

---

## 2. 方法

### 2.1 概览

FlowIPO的核心机制：

1. **在线Rollout**：当前策略与环境交互，收集轨迹和奖励
2. **参考动作生成**：参考策略（EMA更新）在同一状态下生成参考动作
3. **信用分配**：通过当前动作与参考动作的偏移量，结合episode奖励，计算每步的权重
4. **插值Target构造**：在当前动作的velocity target和参考策略的velocity之间做加权插值
5. **标准Flow Matching训练**：用插值后的target做MSE loss更新

### 2.2 数据收集

#### 环境Rollout

使用当前策略 $v_{\theta_{\text{old}}}$ 与环境交互。每个环境步 $i$，策略接收状态 $s_i$（RGB图像 + 语言指令 + proprioception），通过K步去噪生成action chunk $a_i$，执行后得到下一状态 $s_{i+1}$。

Episode结束后获得奖励 $R$：
- Binary reward：$R \in \{0, 1\}$（LIBERO）
- Shaped reward：$R \in \{0, 0.1, 1.0\}$（ManiSkill：0.1 for grasp, 1.0 for place）

**关键**：Rollout时存储每步的初始噪声 $A_i^0$。

#### 参考动作生成

对每个状态 $s_i$，用参考策略 $v_{\text{ref}}$ 从**同一个初始噪声** $A_i^0$ 出发做ODE积分：

$$a_i^{\text{ref}} = \text{ODE\_integrate}(v_{\text{ref}}, A_i^0, s_i)$$

使用同一初始噪声确保 $\delta_i = \|a_i - a_i^{\text{ref}}\|$ 纯粹反映两个策略velocity field的差异，不受噪声随机性干扰。

参考策略通过EMA持续更新（而非frozen SFT）：

$$\theta_{\text{ref}} \leftarrow \beta_m \cdot \theta_{\text{ref}} + (1 - \beta_m) \cdot \theta$$

### 2.3 Step-wise信用分配

#### 核心信号：策略偏移量

对每个环境步 $i$，计算当前动作与参考动作的偏移：

$$\delta_i = \|a_i - a_i^{\text{ref}}\|_2$$

**含义**：$\delta_i$ 大 = 当前策略在这一步做了与参考策略截然不同的决策。

#### 信用分配逻辑

结合episode reward $R$ 和偏移量 $\delta_i$：

| 情况 | 含义 | 应该怎么做 |
|------|------|-----------|
| 成功 + 大偏移 | 关键创新决策导致成功 | **强化**这个动作 |
| 成功 + 小偏移 | 和参考差不多，非关键步 | 温和强化 |
| 失败 + 大偏移 | 关键偏离导致失败 | **回退**到参考策略 |
| 失败 + 小偏移 | 和参考差不多也失败了 | 轻微回退 |

#### 权重计算

Episode内归一化偏移量：

$$A_i = (2R - 1) \cdot \frac{\delta_i - \text{mean}(\{\delta_j\}_{j=0}^{H-1})}{\text{std}(\{\delta_j\}_{j=0}^{H-1}) + \epsilon}$$

转换为插值权重：

$$w_i = \sigma(\alpha \cdot A_i) \in (0, 1)$$

其中 $\sigma$ 是sigmoid函数，$\alpha$ 是温度超参数控制权重的锐度。

#### 扩展：混合信用分配（Shaped Reward环境）

当环境提供step-level reward时，可以混合利用：

$$\delta_i^{\text{mixed}} = \lambda \cdot \frac{\delta_i^{\text{action}} - \text{mean}}{\text{std}} + (1-\lambda) \cdot \frac{\Delta r_i - \text{mean}}{\text{std}}$$

其中 $\Delta r_i = r_{i+1} - r_i$ 是reward差分。信用分配模块是**可插拔的**。

### 2.4 Velocity Target插值

对每个训练样本 $(s_i, a_i, w_i)$：

**Step 1**：构造中间状态（标准flow matching过程）

$$x_t = (1 - t) \cdot a_i + t \cdot \epsilon, \quad t \sim U[0.2, 0.8], \quad \epsilon \sim \mathcal{N}(0, I)$$

> 注：$t$的采样偏向中间区间。$t$接近0时$x_t \approx a_i$，参考策略在此处的预测可能不可靠（偏离参考策略的训练分布）；$t$接近1时$x_t \approx \epsilon$，区分度低。

**Step 2**：计算两个velocity target

当前动作的目标velocity：
$$u_i = a_i - \epsilon$$

参考策略在同一$x_t$上的velocity预测：
$$v_{\text{ref},i} = v_{\text{ref}}(x_t, t, s_i) \quad \text{(一次forward pass，无需梯度)}$$

**Step 3**：插值

$$\tilde{v}_i = w_i \cdot u_i + (1 - w_i) \cdot v_{\text{ref},i}$$

**语义**：
- $w_i \to 1$（成功 + 大偏移）：target ≈ 当前动作的velocity → 强化创新决策
- $w_i \to 0$（失败 + 大偏移）：target ≈ 参考策略的velocity → 回退到安全行为
- $w_i \approx 0.5$（信号弱）：target ≈ 两者均值 → 温和更新

### 2.5 训练损失

$$\mathcal{L}(\theta) = \mathbb{E}_{i, t, \epsilon}\left[\|v_\theta(x_t, t, s_i) - \tilde{v}_i\|^2\right]$$

**就是标准的flow matching MSE loss，唯一的区别是target从固定的 $u_i$ 变成了动态的 $\tilde{v}_i$。**

无需任何额外的loss项、正则项（KL正则已隐式包含在插值机制中）。

---

## 3. 理论分析

### 3.1 Mirror Descent等价性

**Theorem 1 (Interpolation Optimality)**

$\tilde{v}_i = w_i \cdot u_i + (1 - w_i) \cdot v_{\text{ref}}$ 是如下加权最近邻问题的解析解：

$$\tilde{v}_i = \arg\min_v \left[w_i \|v - u_i\|^2 + (1 - w_i)\|v - v_{\text{ref}}\|^2\right]$$

**证明**：对$v$求导并令其为零：$2w_i(v - u_i) + 2(1-w_i)(v - v_{\text{ref}}) = 0$，解得 $v = w_i \cdot u_i + (1-w_i) \cdot v_{\text{ref}}$。$\square$

**含义**：FlowIPO等价于在velocity space中做mirror descent——在拟合当前动作和回退到参考策略之间寻找最优平衡，平衡系数由step-level信用分配决定。

### 3.2 Loss分解

**Theorem 2 (Implicit KL Regularization)**

FlowIPO的loss可以分解为三项：

$$\|v_\theta - \tilde{v}_i\|^2 = \|v_\theta - v_{\text{ref}}\|^2 - 2w_i\langle v_\theta - v_{\text{ref}},\; u_i - v_{\text{ref}}\rangle + w_i^2\|u_i - v_{\text{ref}}\|^2$$

**证明**：将 $\tilde{v}_i = v_{\text{ref}} + w_i(u_i - v_{\text{ref}})$ 代入展开即得。$\square$

各项含义：
- 第一项 $\|v_\theta - v_{\text{ref}}\|^2$：**KL正则**——拉向参考策略，防止策略崩溃
- 第二项 $-2w_i\langle v_\theta - v_{\text{ref}},\; u_i - v_{\text{ref}}\rangle$：**方向性改进信号**——$w_i$加权地引导velocity偏向成功动作的方向
- 第三项 $w_i^2\|u_i - v_{\text{ref}}\|^2$：不含$\theta$的常数，不影响梯度

**关键对比**：π-StepNFT的wMSE分解（Theorem 4.5）包含一个**隐式分离惩罚** $\|d_t\|^2$，抑制策略更新。FlowIPO的分解中**没有这样的惩罚项**——KL正则和方向性信号是解耦的。

### 3.3 单调安全性

**Theorem 3 (Monotonic Safety Guarantee)**

设参考策略的期望回报为 $J_{\text{ref}}$，当前策略为 $J_{\theta}$。在soft update（EMA）和准确信用分配的假设下，FlowIPO的更新不会使策略退化到 $J_{\text{ref}}$ 以下。

**直觉**：最坏情况下（所有$w_i = 0$），FlowIPO完全回退到参考策略。由于参考策略通过EMA更新，其性能单调不降。因此FlowIPO提供了一个"安全地板"。

### 3.4 隐式负信号

**Theorem 4 (Implicit Negative Gradient)**

对于失败样本（$R=0$, $w_i < 0.5$），FlowIPO的梯度包含一个隐式的"推离"分量：

$$-\nabla_\theta \mathcal{L} \propto -(v_\theta - v_{\text{ref}}) - w_i(u_i^{\text{fail}} - v_{\text{ref}}) \cdot \nabla_\theta v_\theta$$

其中第一项将策略拉向参考（安全回退），第二项中 $u_i^{\text{fail}} - v_{\text{ref}}$ 是失败动作偏离参考的方向，乘以小权重$w_i$，实际效果是**策略被推离失败动作方向**。

这意味着FlowIPO无需显式构造负样本或镜像分支，即可自然获得负信号。

---

## 4. 与现有方法的关系与区别

### 4.1 对比总结

| 维度 | πRL (PPO) | πRL (GRPO) | RFT | DiffusionNFT/π-StepNFT | **FlowIPO** |
|------|-----------|------------|-----|-------------------------|-------------|
| 需要似然 | ✅ | ✅ | ❌ | ❌ | **❌** |
| 需要Critic | ✅ | ❌ | ❌ | ❌ | **❌** |
| 需要镜像构造 | ❌ | ❌ | ❌ | ✅ | **❌** |
| 失败样本利用 | ✅ (A<0) | ✅ (norm后为负) | ❌ (丢弃) | ✅ (负分支) | **✅ (回退参考)** |
| Step-level信号 | ✅ (GAE) | ❌ | ❌ | ❌ (episode-level) | **✅ ($\delta_i$权重)** |
| 负信号方向 | 由advantage定 | 由group norm定 | 无 | 推离（方向不确定） | **回退参考（方向确定）** |
| 额外网络 | Critic (+noise net) | 无 | 无 | 无 | **无** |
| 额外计算 | ~2x | ~Nx (N条rollout) | ~1x | ~1.3x | **~1.5x** |
| OOD鲁棒性 | 差（critic过拟合） | 中 | 中 | 好 | **好（无critic）** |

### 4.2 与RFT的关键区别

RFT（Rejection Fine-Tuning）只在成功样本上做flow matching，丢弃失败样本。DiffusionNFT的消融已明确显示：**去掉负信号会导致reward collapse**。

FlowIPO对失败样本有明确处理——target被设为参考策略的velocity，模型被引导回退到安全行为。这提供了有方向的负信号（回到参考），比NFT的负信号（离开当前，方向不定）更稳定。

### 4.3 与DPO/Diffusion-DPO的关系

DPO需要配对偏好数据和似然比。FlowIPO不需要配对数据，不需要似然计算。两者在数学形式上完全不同。

### 4.4 方法定位

FlowIPO的核心范式转变：

> **从"把flow模型塞进RL框架"到"把RL信号注入flow matching目标"**

这避开了似然估计、value estimation等所有RL-for-flow的核心难题，回归到最朴素的supervised learning形式，但target是由在线交互结果动态决定的。

---

## 5. 算法伪代码

**Algorithm: FlowIPO**

---

**Input:**
- SFT-initialized flow VLA: $v_\theta$
- Reference policy: $v_{\mathrm{ref}} \leftarrow \mathrm{copy}(v_\theta)$（初始化为SFT权重）
- Rollout policy: $v_{\theta_{\mathrm{old}}} \leftarrow \mathrm{copy}(v_\theta)$
- Hyperparams: $\alpha$（temperature），$\beta_{\mathrm{ref}},\, \beta_{\mathrm{old}}$（EMA rates）

---

**For each iteration** $m = 1, 2, \ldots, M$:

**▷ Phase 1: Online Rollout**

$D \leftarrow \emptyset$

For each task（initial state $s_0$, language prompt $c$）：

- For $i = 0$ to $H-1$：
  - Sample initial noise: $A_i^0 \sim \mathcal{N}(0,\, I)$
  - $K$-step ODE denoising: $a_i = \mathrm{ODE\text{-}Integrate}(v_{\theta_{\mathrm{old}}},\, A_i^0,\, s_i,\, c)$
  - Execute $a_i$ in environment $\rightarrow s_{i+1}$；store $(s_i,\, c,\, a_i,\, A_i^0)$
- Observe episode reward $R$
- For $i = 0$ to $H-1$（frozen, no grad）：
  - $a_i^{\mathrm{ref}} = \mathrm{ODE\text{-}Integrate}(v_{\mathrm{ref}},\, A_i^0,\, s_i,\, c)$

$$D \leftarrow D \;\cup\; \bigl\{(s_i,\; c,\; a_i,\; a_i^{\mathrm{ref}},\; A_i^0,\; R)\bigr\}_{i=0}^{H-1}$$

**▷ Phase 2: Step-wise Credit Assignment**

For each episode in $D$：

- Per-step policy divergence: $\delta_i = \|a_i - a_i^{\mathrm{ref}}\|_2$
- Normalized credit signal: $\displaystyle A_i = (2R - 1) \cdot \frac{\delta_i - \mathrm{mean}(\{\delta_j\})}{\mathrm{std}(\{\delta_j\}) + \varepsilon}$
- Interpolation weight: $w_i = \sigma(\alpha \cdot A_i) \in (0,\, 1)$

**▷ Phase 3: Interpolated Target Construction**

For each sample $(s_i,\, c,\, a_i,\, w_i)$ in batch：

- Sample $t \sim U[0.2,\, 0.8]$，$\varepsilon \sim \mathcal{N}(0, I)$；construct $x_t = (1-t)\cdot a_i + t \cdot \varepsilon$
- Current target: $u_i = a_i - \varepsilon$
- Reference velocity: $v_{\mathrm{ref},i} = v_{\mathrm{ref}}(x_t,\, t,\, s_i,\, c)$（no grad）
- Interpolated target: $\tilde{v}_i = w_i \cdot u_i + (1 - w_i) \cdot v_{\mathrm{ref},i}$

**▷ Phase 4: Flow Matching Update**

$$\mathcal{L}(\theta) = \mathbb{E}_{i,\,t,\,\varepsilon}\!\left[\bigl\|v_\theta(x_t,\, t,\, s_i,\, c) - \tilde{v}_i\bigr\|^2\right]$$

$$\theta \leftarrow \theta - \mathrm{lr} \cdot \nabla_\theta \mathcal{L}$$

**▷ Phase 5: Policy Synchronization**

- Update rollout policy: $\theta_{\mathrm{old}} \leftarrow \beta_{\mathrm{old}} \cdot \theta_{\mathrm{old}} + (1 - \beta_{\mathrm{old}}) \cdot \theta$
- Update reference policy（$\beta_{\mathrm{ref}} > \beta_{\mathrm{old}}$，更新更慢）: $\theta_{\mathrm{ref}} \leftarrow \beta_{\mathrm{ref}} \cdot \theta_{\mathrm{ref}} + (1 - \beta_{\mathrm{ref}}) \cdot \theta$

---

**Output:** Optimized policy $v_\theta$

---

## 6. 实现细节

### 6.1 架构

- **Base model**：π0 (PaliGemma 3B VLM + 300M flow action expert) 或 π0.5
- **RL阶段冻结VLM**，只微调action expert（~300M参数），与πRL一致
- **参考模型**：action expert部分的EMA副本，VLM共享（frozen）

### 6.2 Rollout

- Rollout和参考动作生成均使用**ODE采样**，保证 $\delta_i = \|a_i - a_i^{\text{ref}}\|$ 纯粹反映policy divergence，不受SDE中间步随机噪声污染
- 探索多样性来源于每步独立采样的初始噪声 $A_i^0 \sim \mathcal{N}(0,I)$，无需SDE引入额外随机性
- ODE采样用于评估
- 生成参考动作时使用与rollout相同的初始噪声 $A^0$
- Rollout时只需额外存储 $A_i^0$（一个向量），内存开销可忽略

### 6.3 关键超参数

| 超参数 | 含义 | 建议范围 | 默认值 |
|--------|------|---------|--------|
| $\alpha$ | 插值权重温度 | [0.5, 5.0] | 2.0 |
| $\beta_{\text{ref}}$ | 参考策略EMA率 | [0.99, 0.999] | 0.995 |
| $\beta_{\text{old}}$ | Rollout策略EMA率 | [0.9, 0.99] | 0.95 |
| $t$ range | Flow matching time采样范围 | [0.1, 0.9] | U[0.2, 0.8] |
| K | 去噪步数 | 3-8 | 4 |
| H | Action chunk size | 5-10 | 5 (short) / 10 (long) |

### 6.4 计算开销分析

每个训练迭代的额外开销（相比标准flow matching SFT）：

| 操作 | 开销 | 说明 |
|------|------|------|
| 参考模型ODE积分 | +0.3x | K=4步forward pass，无梯度 |
| 参考模型单步velocity | +0.1x | 训练时一次forward，无梯度 |
| $\delta_i, w_i$ 计算 | 可忽略 | 标量运算 |
| 总额外开销 | **~1.4-1.5x** | 远低于πRL的~2x（需训练critic）|

---

## 7. 实验设计

### 7.1 Benchmarks

| Benchmark | 任务描述 | 奖励类型 | 核心测试 |
|-----------|---------|---------|---------|
| LIBERO (4 suites) | 4类操控任务，各10子任务 | Binary (0/1) | Few-shot SFT + RL性能 |
| ManiSkill | 4352种pick-and-place组合 | Shaped (0/0.1/1.0) | IND性能 + OOD泛化 |

### 7.2 Baselines

| 方法 | 类型 | 来源 |
|------|------|------|
| SFT (few-shot) | 纯监督 | 基线 |
| πRL (Flow-SDE + PPO) | 似然+Critic | πRL论文 |
| πRL (Flow-SDE + GRPO) | 似然+组排序 | πRL论文 |
| πRL (Flow-Noise + PPO) | 似然+Critic | πRL论文 |
| RFT | 成功样本SFT | 基线方法 |
| **FlowIPO (ours)** | **Velocity插值** | **本文** |

### 7.3 主实验

#### LIBERO (Few-shot设定)

预期结果：

| 方法 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| SFT (π0) | 65.3 | 64.4 | 49.8 | 51.2 | 57.6 |
| πRL (PPO) | 98.4 | 99.4 | 96.2 | 90.2 | 96.1 |
| πRL (GRPO) | 97.8 | 97.8 | 83.2 | 81.4 | 90.0 |
| **FlowIPO** | ~97 | ~99 | ~93 | ~87 | ~94 |

预期：超越GRPO，接近或竞争PPO。在Goal/Long任务上，step-level信用分配（$\delta_i$）应该提供有价值的信号。

#### ManiSkill (IND + OOD)

预期结果：

| 方法 | IND | OOD Vision | OOD Semantic | OOD Execution | OOD Avg. |
|------|-----|-----------|-------------|--------------|----------|
| SFT (π0) | 38.4 | 32.6 | 8.4 | 13.2 | 18.1 |
| πRL (PPO) | 78.8 | 61.1 | 25.4 | 31.5 | 39.3 |
| **FlowIPO** | ~76 | ~65 | ~35 | ~32 | ~44 |

预期：IND略低于PPO（无critic的代价），但OOD显著优于PPO（无critic → 不过拟合视觉特征）。

### 7.4 消融实验

#### 消融1：信用分配信号

| 变体 | 描述 |
|------|------|
| FlowIPO (full) | $w_i$ 基于 $\delta_i$ 和 $R$ |
| FlowIPO (uniform) | $w_i = R$（所有步相同权重） |
| FlowIPO (random) | $w_i$ 随机 |

验证step-level信用分配是否带来实质提升。

#### 消融2：插值 vs Binary

| 变体 | 描述 |
|------|------|
| FlowIPO (continuous) | $w_i \in (0, 1)$ 连续插值 |
| FlowIPO (binary) | $w_i \in \{0, 1\}$ 硬选择 |
| RFT only | $w_i = R$，失败样本丢弃 |

验证连续插值和利用失败样本的价值。

#### 消融3：参考策略

| 变体 | 描述 |
|------|------|
| EMA reference | $\theta_{\text{ref}}$ 通过EMA更新 |
| Frozen SFT reference | $\theta_{\text{ref}}$ 固定为初始SFT |

验证EMA参考策略的必要性。

#### 消融4：温度 $\alpha$ 敏感度

$\alpha \in \{0.5, 1.0, 2.0, 5.0\}$

$\alpha$过小 → 所有$w_i$接近0.5 → 信号弱
$\alpha$过大 → $w_i$趋向0或1 → 退化为binary选择

#### 消融5：Time采样范围

$t \sim U[0, 1]$ vs $U[0.2, 0.8]$ vs $U[0.3, 0.7]$

验证避开边界$t$值的必要性。

### 7.5 分析实验

- **$\delta_i$的分布可视化**：成功轨迹 vs 失败轨迹各步的$\delta_i$分布
- **$w_i$与任务关键步的对应关系**：在LIBERO中，关键步（如抓取、放置瞬间）的$w_i$是否更极端
- **训练曲线对比**：收敛速度、稳定性
- **Episode length分析**：类似πRL的Fig 13

---

## 8. 潜在风险与缓解

### 8.1 信用分配信号区分度不足

**风险**：如果RL策略在所有步上均匀偏离参考，$\delta_i$无区分度，$w_i$退化为接近uniform。

**缓解**：
1. 此时FlowIPO退化为 "成功样本做FM + 失败样本做蒸馏"的软RFT，仍然比纯RFT好（失败样本没被丢弃）
2. 在shaped reward环境（ManiSkill），混合reward差分信号
3. 用EMA参考策略，随着训练推进偏移自然增大

**验证**：先在LIBERO-Spatial上做最小实验，检查$\delta_i$分布。

### 8.2 参考策略质量

**风险**：Few-shot SFT模型成功率仅57-77%，回退到它不一定好。

**缓解**：
1. EMA参考策略随训练变好
2. 回退是soft的（$w_i \in (0.3, 0.5)$而非0），不完全回退
3. $w_i$受$\delta_i$调制——失败但偏移小的步$w_i \approx 0.5$，几乎不回退

### 8.3 参考策略在OOD $x_t$上预测不准

**风险**：当$t$小时，$x_t \approx a_i$（失败动作），偏离参考策略分布。

**缓解**：$t$采样偏向中高区间$[0.2, 0.8]$，高$t$时$x_t \approx \epsilon$与动作无关。

### 8.4 审稿人可能的challenge

**Q: "这还算RL吗？"**
A: 满足RL核心要素——在线交互、试错、策略改进。定位为"reward-guided online flow matching"，这是一个比policy gradient更适合flow模型的优化范式。

**Q: "和RFT的区别不够大？"**
A: 三个关键区别——(1) 失败样本有明确方向的处理，(2) step-level信用分配，(3) 连续插值而非binary。消融实验逐一验证每个组件的贡献。

**Q: "$\delta_i$作为信用分配的理论依据？"**
A: $\delta_i$度量的是当前策略相对参考策略的行为偏移。结合episode outcome，它识别"导致成功/失败的关键偏离步"。理论上，这是策略差异对回报贡献的一阶近似。

---

## 9. 论文叙事结构

### Abstract
Flow-based VLAs面临RL似然不可解难题。我们提出FlowIPO，完全绕过似然估计，通过在velocity target上做reward-directed插值实现策略改进。无需似然、无需Critic、无需额外网络。在LIBERO和ManiSkill上验证有效性。

### Introduction
1. VLA + RL的重要性（打破SFT的数据瓶颈）
2. Flow-based VLA的似然困难（πRL的两种方案都很重）
3. 我们的洞察：不走PG路线，直接在velocity target上注入RL信号
4. FlowIPO的三个特点：无似然、有step-level信用分配、有安全回退

### Method
1. Preliminaries (flow matching for VLA)
2. Data collection & reference action generation
3. Step-wise credit assignment via policy divergence
4. Interpolated velocity target construction
5. Training objective (standard MSE)
6. Theoretical analysis (mirror descent equivalence, implicit negative signal, safety guarantee)

### Experiments
1. LIBERO main results
2. ManiSkill main results + OOD generalization
3. Ablations (credit assignment, interpolation, reference policy, hyperparameters)
4. Analysis (δ distribution, training dynamics)

### Conclusion
FlowIPO证明了flow-based VLA的RL不需要复杂的似然估计machinery。通过把RL信号直接注入flow matching的训练目标，以最简的形式实现了有竞争力的策略改进。

---

## 10. 代码改动量估算

基于πRL的开源代码（RLinf框架），FlowIPO的核心改动：

| 模块 | 改动 | 行数估算 |
|------|------|---------|
| Rollout | 存储$A^0$，计算$a^{\text{ref}}$ | ~30行 |
| Credit assignment | 计算$\delta_i$, $A_i$, $w_i$ | ~15行 |
| Target construction | 插值$\tilde{v}_i$ | ~10行 |
| Loss function | 替换target | ~5行 |
| EMA reference | 参考策略更新 | ~10行 |
| **总计** | | **~70行核心代码** |

无需：似然计算模块、Critic网络、PPO objective、GAE computation。

实际上是在简化代码，删掉的比加上的多。