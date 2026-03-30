# Flow Policy Iteration: Reinforcement Learning as Distribution Transport

## NeurIPS 2026 完整研究方案（Final Version）

---

## 一、Motivation：一个根本性矛盾

### 1.1 现有范式的共同错误

截至2026年2月，Flow Matching + RL领域已有15+篇论文。它们**全部**共享同一个思路：

> "Flow matching是好的policy class → 如何把RL算法嫁接上去？"

这导致每篇论文都在与同一组技术难题搏斗：

| 技术难题 | 本质原因 | 谁在挣扎 |
|---------|---------|---------|
| Log-likelihood不可解 | Flow matching没有closed-form log π(a\|s)，但PPO/SAC需要 | FPO(用CFM loss做proxy), π_RL(加noise network), SAC Flow(重参数化velocity) |
| Backprop through ODE不稳定 | Multi-step ODE反传等价于RNN → 梯度爆炸/消失 | SAC Flow(GRU重参数化), RFO(gradient clipping), QAM(adjoint绕过) |
| Objective mismatch | Flow matching目标(velocity MSE) vs RL目标(max reward) | FlowRL(W₂正则化), ORW-CFM(W₂), Decision Flow(重定义flow step) |
| 多模态collapse | RL gradient update倾向collapse到单一mode | ORW-CFM(Lemma 1证明不加正则必然collapse), FlowRL(W₂约束) |

**根本原因**：这些矛盾全部源于一个错误前提——**把Flow Matching当作policy class，用外部RL算法优化它**。这就像用锤子（RL算法）敲螺丝（Flow Matching），然后不断改良锤子。

### 1.2 核心Insight

**Policy Improvement本质上是一个distribution transport问题。Flow Matching天然就是在做distribution transport。**

KL-regularized policy improvement的closed-form解：

$$\pi_{k+1}(a \mid s) \propto \pi_k(a \mid s) \cdot \exp\!\bigl(A^{\pi_k}(s, a) / \lambda\bigr)$$

这定义了从 $\pi_k$（source）到 $\pi_{k+1}$（target）的distribution transport。Flow matching可以直接学习这个transport——不需要likelihood、不需要policy gradient、不需要backprop through ODE。

**一句话message**：*The optimal way to do RL with flow matching is... just flow matching — with the right weights.*

---

## 二、相关工作全景（截至2026.02）

### 2.1 On-Policy Policy Gradient路线

| 论文 | 时间 | 核心方法 | Venue |
|------|------|---------|-------|
| FPO (McAllister et al., Berkeley) | 2025.07 | CFM loss比值近似likelihood ratio，嵌入PPO-clip | Preprint |
| FPO for VLA (Lyu et al.) | 2025.10 | 针对π₀的FPO + structure-aware credit assignment | Preprint |
| ReinFlow (Zhang et al.) | 2025.05 | 可学习noise network实现精确log-likelihood | NeurIPS 2025 |
| π_RL (Chen/Zhang et al.) | 2025.10 | Flow-Noise + Flow-SDE，支持π₀/π₀.₅ | Preprint |
| Flow-GRPO (Liu et al.) | 2025.05 | ODE→SDE + GRPO | Preprint |
| RFO | 2026.01 | Reparameterization trick直接反传 + CFM正则化 | Preprint |

### 2.2 Off-Policy / Actor-Critic路线

| 论文 | 时间 | 核心方法 | Venue |
|------|------|---------|-------|
| FlowRL | 2025.09 | W₂正则化的flow policy + Q值联合优化 | NeurIPS 2025 |
| QAM | 2026.01 | Adjoint Matching + TD Q-learning | Preprint |
| SAC Flow | 2025.09 | Flow rollout≡RNN → GRU/Transformer重参数化 | Preprint |
| OFQL | 2026.02 | Average velocity field → one-step action | Preprint |

### 2.3 黑盒 / Latent Steering路线

| 论文 | 时间 | 核心方法 | Venue |
|------|------|---------|-------|
| DSRL (Wagenmaker et al.) | 2025.06 | 冻结policy权重，latent noise空间SAC | CoRL 2025 |

### 2.4 Reward-Weighted / SOC路线

| 论文 | 时间 | 核心方法 | Venue |
|------|------|---------|-------|
| Adjoint Matching | 2024.09 | Memoryless SOC + regression | ICLR 2025 Spotlight |
| ORW-CFM-W₂ | 2024.10 | Online reward-weighted CFM + W₂防collapse | ICLR 2025 |
| RWFM (Pfrommer et al.) | 2025.07 | 单轮reward-weighted flow matching | Preprint |
| AWM (Xue et al.) | 2025.09 | Advantage weighted matching | Preprint |

### 2.5 World Model路线

| 论文 | 时间 | 核心方法 | Venue |
|------|------|---------|-------|
| VLAW | 2026.02 | World model rollout + VLM reward model + supervised PI | Preprint |
| WoVR | 2026.02 | World model作simulator, RL fine-tune VLA | Preprint |

### 2.6 定位总结

所有上述方法分为两大阵营：
- **阵营A**（FPO, π_RL, ReinFlow, SAC Flow, FlowRL, QAM, DSRL, RFO）：把RL算法嫁接到flow policy上 → 与likelihood/gradient难题搏斗
- **阵营B**（RWFM, ORW-CFM, AWM）：用reward weighting替代RL → 简单但缺乏iterative improvement理论

**FPI = 阵营B的简洁性 + 阵营A的理论保证 + 统一框架。**

---

## 三、方法：Flow Policy Iteration (FPI)

### 3.1 Problem Setup

**环境**：MDP $(S, A, P, R, \gamma)$，$A \subseteq \mathbb{R}^{d_a}$ 连续action space。

**初始策略**：$\pi_0$ = pre-trained flow matching VLA（如π₀/π₀.₅），velocity field $v_{\theta_0}(x_\tau, \tau \mid o, \ell)$，其中 $o$ 是observation，$\ell$ 是language instruction。

**目标**：通过iterative improvement得到 $\pi_K$，最大化任务成功率。

### 3.2 Value Estimation（采用π_RL方案）

训练一个value network $\mathcal{V}_\phi(o, \ell)$，评估从observation $o$ 出发、执行instruction $\ell$ 的预期成功程度。

#### Stage 1：Progress Warm-Start

用成功demonstration数据 $\mathcal{D}_{\text{exp}}$ 暖启动value network：

$$\mathcal{L}_{\text{prog}} = \mathbb{E}_{(o_t, \ell) \sim \mathcal{D}_{\text{exp}}} \left[ \left( \mathcal{V}_\phi(o_t, \ell) - \frac{t}{T} \right)^2 \right]$$

其中 $t$ 是episode内时间步，$T$ 是总长度。直觉：成功trajectory的进度近似线性增长，$t/T$ 提供免费的dense supervision。这让 $\mathcal{V}$ 从一开始就有合理的temporal structure。

#### Stage 2：TD Fine-Tuning

在online rollout数据 $\mathcal{D}$ 上用TD learning精修：

$$\mathcal{L}_{\text{TD}} = \mathbb{E}_{(o_t, \ell, o_{t+1}) \sim \mathcal{D}} \left[ \left( \mathcal{V}_\phi(o_t, \ell) - y_t \right)^2 \right]$$

$$y_t = r_t + \gamma \, \mathcal{V}_{\bar{\phi}}(o_{t+1}, \ell)$$

其中 $r_t = 0$（中间步），$r_T = +1$（成功）或 $r_T = -1$（失败），$\bar{\phi}$ 为target network参数。

TD learning的关键作用：学会区分成功和失败——progress prior对failure不敏感（失败trajectory的前半段可能看起来也在"进步"），TD通过bootstrap把terminal的 $-1$ 信号传播回来。

**总loss**：

$$\mathcal{L}_{\mathcal{V}} = \mathcal{L}_{\text{prog}} + \mathcal{L}_{\text{TD}}$$

#### Action-Chunk Advantage Estimation

每步生成action chunk $\mathbf{a}_t = (a_t^1, \ldots, a_t^H)$，执行后观测到一系列observation $o_{t+1}, \ldots, o_{t+H}$。Advantage定义为：

$$A(o_t, \mathbf{a}_t, \ell) = \frac{1}{H} \sum_{k=1}^{H} \mathcal{V}_\phi(o_{t+k}, \ell) \;-\; \mathcal{V}_\phi(o_t, \ell)$$

直觉：执行这个action chunk后，平均value是否提升了。$A > 0$ 说明chunk整体有益，$A < 0$ 说明chunk导致退步。

### 3.3 Policy Improvement via Weighted Flow Matching

#### 核心操作

给定rollout数据集 $\{(o_t, \mathbf{a}_t, A_t)\}$，计算importance weight：

$$w_t = \exp\!\left(\frac{A(o_t, \mathbf{a}_t, \ell)}{\lambda}\right)$$

然后执行加权flow matching更新：

$$\mathcal{L}_{\text{FPI}}(\theta) = \mathbb{E}_{(o, \mathbf{a}, w) \sim \mathcal{D}} \; \mathbb{E}_{\tau \sim U[0,1], \, \epsilon \sim \mathcal{N}(0,I)} \left[ w \cdot \left\| v_\theta(x_\tau, \tau \mid o, \ell) - (\mathbf{a} - \epsilon) \right\|^2 \right]$$

其中 $x_\tau = \tau \cdot \mathbf{a} + (1-\tau) \cdot \epsilon$ 是标准conditional flow matching的插值。

**这就是全部。** Policy improvement = 在advantage-weighted rollout数据上做标准flow matching训练。

#### 直觉解释

- $A > 0$（好的action chunk）→ $w > 1$ → flow matching在这个样本上的loss被放大 → velocity field更努力地学习transport到这类action
- $A < 0$（差的action chunk）→ $w < 1$ → loss被缩小 → velocity field少关注这类action
- $A \approx 0$（平均水平）→ $w \approx 1$ → 等价于standard BC

当 $\lambda \to \infty$ 时，所有 $w \to 1$，退化为BC。当 $\lambda \to 0$ 时，只学advantage最高的action。$\lambda$ 控制exploitation程度。

#### Weight Normalization（实践要点）

为避免importance weight的数值问题，使用self-normalization：

$$\hat{w}_i = \frac{\exp(A_i / \lambda)}{\frac{1}{N}\sum_{j=1}^N \exp(A_j / \lambda)}$$

保证batch内weight的平均值为1。此外clip weight到 $[w_{\min}, w_{\max}]$ 防止极端值（实践中 $w_{\max} = 10$ 即可）。

### 3.4 完整算法

---

#### Algorithm 1: Flow Policy Iteration (FPI)

**Input:** Pre-trained flow VLA $\pi_{\theta_0}$，温度 $\lambda$，迭代轮数 $K$，value network $\mathcal{V}_\phi$

**Pre-training Phase:** 在demonstration数据上用 $\mathcal{L}_{\text{prog}}$ 暖启动 $\mathcal{V}_\phi$

**For** $k = 0, 1, \ldots, K-1$ **do:**

**Step 1 (Rollout):** 用 $\pi_{\theta_k}$ 在 $N_{\text{env}}$ 个并行环境中执行 $M$ 个episodes，收集

$$\mathcal{D}_k = \bigl\{(o_t^{(i)},\; \ell^{(i)},\; \mathbf{a}_t^{(i)},\; r_t^{(i)},\; o_{t+1}^{(i)},\; \ldots,\; o_{t+H}^{(i)})\bigr\}$$

**Step 2 (Value Update):** 在 $\mathcal{D}_k$ 上多步梯度下降更新 $\mathcal{V}_\phi$（TD target使用target network $\bar{\phi}$）：

$$\phi \;\leftarrow\; \phi - \alpha_V \,\nabla_\phi \bigl(\mathcal{L}_{\text{prog}} + \mathcal{L}_{\text{TD}}\bigr)$$

$$\mathcal{L}_{\text{prog}} = \mathbb{E}\!\left[\bigl(\mathcal{V}_\phi(o_t, \ell) - t/T\bigr)^2\right], \qquad \mathcal{L}_{\text{TD}} = \mathbb{E}\!\left[\bigl(\mathcal{V}_\phi(o_t, \ell) - y_t\bigr)^2\right], \quad y_t = r_t + \gamma\,\mathcal{V}_{\bar{\phi}}(o_{t+1}, \ell)$$

其中 $r_t = 0$（中间步），$r_T = +1$（成功）/ $-1$（失败）。

**Step 3 (Advantage & Weight):** 对 $\mathcal{D}_k$ 中每个action chunk $(o_t, \mathbf{a}_t)$ 计算：

$$A_t = \frac{1}{H}\sum_{j=1}^{H} \mathcal{V}_\phi(o_{t+j},\, \ell) \;-\; \mathcal{V}_\phi(o_t,\, \ell)$$

$$w_t = \operatorname{clip}\!\left(\frac{\exp(A_t / \lambda)}{\frac{1}{N}\sum_{i=1}^{N}\exp(A_i / \lambda)},\;\; w_{\min},\;\; w_{\max}\right)$$

**Step 4 (Weighted Flow Matching Update with Trust Region):** 预计算 $(\tau, \epsilon)$ 和trust region anchor $v_{\text{old}} = v_{\bar{\theta}}(x_\tau, \tau)$（EMA模型）。在 $\mathcal{D}_k$ 上多步梯度下降更新 $\theta$：

$$\mathcal{L}_{\text{FPI}}(\theta) = \mathbb{E}_{(o,\mathbf{a},w)} \!\left[\; w \cdot \left\| v_\theta\!\left(x_\tau,\, \tau \mid o,\, \ell\right) - (\mathbf{a} - \epsilon) \right\|^2 + \beta_{\text{kl}} \cdot \left\| v_\theta(x_\tau, \tau) - v_{\text{old}}(x_\tau, \tau) \right\|^2 \;\right]$$

其中 $x_\tau = \tau \cdot \mathbf{a} + (1-\tau) \cdot \epsilon$，$(\tau, \epsilon, v_{\text{old}})$ 在rollout时预计算并固定（多个update epoch共用）。

$$\theta_{k+1} \;\leftarrow\; \theta_k - \alpha_\pi \,\nabla_\theta \,\mathcal{L}_{\text{FPI}}(\theta)$$

**End For**

**Return** $\pi_{\theta_K}$

---

---

### 3.5 关键实现细节

#### 兼容π₀/π₀.₅架构

π₀的结构是 VLM encoder → flow matching action head。FPI只修改flow matching action head的训练方式——从uniform loss变为weighted loss。VLM encoder可以冻结或joint fine-tune（推荐冻结 + LoRA）。

#### 与LoRA的天然兼容

因为FPI的loss就是标准MSE（加了weight），所有parameter-efficient fine-tuning技术（LoRA, QLoRA, adapter）直接适用：

$$\mathcal{L}_{\text{FPI}}(\Delta\theta) = \mathbb{E}\left[ w \cdot \left\| v_{\theta_0 + \Delta\theta}(x_\tau, \tau \mid o, \ell) - \text{target} \right\|^2 \right]$$

不像FPO/SAC Flow需要特殊的gradient computation pipeline。

#### Off-Policy数据利用与Trust Region

FPI的weighted loss是regression形式，理论上支持多个epoch的gradient update。但实践中发现**纯FPI在多个update epoch下会collapse**——velocity field在每轮训练中逐渐漂移，4轮后累积偏移导致下一次rollout的action分布严重偏离训练分布。

**解决方案**：添加trust region正则化，预计算 $v_{\text{old}} = v_{\bar{\theta}}(x_t, t)$（EMA模型的velocity），训练时加KL penalty：

$$\mathcal{L}_{\text{FPI-TR}}(\theta) = \mathbb{E}\left[ w \cdot \|v_\theta - u\|^2 + \beta_{\text{kl}} \cdot \|v_\theta - v_{\text{old}}\|^2 \right]$$

其中 $(t, \epsilon)$ 在rollout worker上预计算并固定——同一批数据的4个update epoch使用相同的 $(t, \epsilon, v_{\text{old}})$。这保证：
1. $v_{\text{old}}$ 是有效的比较基准（相同 $(x_t, t)$ 下的velocity）
2. KL penalty阻止velocity field偏离太远
3. 多epoch训练稳定，不会collapse

#### Value Network实践要点

Progress Warm-Start在实践中需要调整：π-RL原方案在demo数据上预训练V，但我们没有单独的demo数据加载器。替代方案：**reward-scaled progress target**：

$$p_t = \frac{t}{T} \cdot R$$

其中 $R$ 是episode reward（成功=1，失败=0）。这样：
- 成功轨迹：target = $t/T$（标准progress prior）
- 失败轨迹：target = $0$（value应该始终低）

所有轨迹都贡献supervision，近似π-RL demo预训练的效果。

此外，必须设置 `detach_critic_input: True`——阻止value loss梯度穿透VLM backbone，否则会腐蚀shared features导致policy退化。

#### Advantage标准化

直接对raw advantage做 $\exp(A/\lambda)$ 会随着value network改善导致weight爆炸。解决方案：先标准化advantage到zero-mean unit-variance，再做exponential tilting：

$$\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}, \quad w_t = \frac{\exp(\hat{A}_t / \lambda)}{\frac{1}{N}\sum_j \exp(\hat{A}_j / \lambda)}$$

#### Value Warm-Up

前 $N$ 个epoch使用 $w=1$（纯BC），只训练value head。等value head学到有意义的temporal structure后，再启用advantage weighting。推荐 $N = 10$。

#### 温度调度

推荐 $\lambda$ 随训练进展递减（类似simulated annealing）：
- 早期 $\lambda$ 大 → 接近BC → 保持exploration
- 后期 $\lambda$ 小 → 更aggressive地upweight good actions → 更快improvement

具体schedule：$\lambda_k = \lambda_0 \cdot \beta^k$，$\beta \in [0.9, 0.99]$。

---

## 四、理论分析

### 4.1 Theorem 1: Policy Improvement Guarantee

**定理1**（Monotonic Improvement）.
设 $\pi_{k+1}$ 由Algorithm 1产生，flow matching的fitting error满足 $\mathbb{E}_s[W_2(\hat{\pi}_{k+1}(\cdot|s), \pi_{k+1}^*(\cdot|s))] \leq \epsilon_{\text{FM}}$，advantage estimation error满足 $\|\hat{A} - A^{\pi_k}\|_\infty \leq \epsilon_A$。则：

$$V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s) - \frac{2\gamma}{(1-\gamma)^2}\left(L_Q \cdot \epsilon_{\text{FM}} + \epsilon_A\right)$$

其中 $L_Q$ 是Q函数的Lipschitz常数。

**含义**：只要 (1) flow matching拟合足够好，(2) advantage估计足够准，每步FPI严格improve。两个error项的结构表明，FPI的性能同时受policy improvement quality和value estimation quality制约——advantage estimation不是被忽略的问题，而是被明确纳入了理论bound。

**证明思路**：
1. Exact KL-regularized PI: $V^{\pi_{k+1}^*} \geq V^{\pi_k}$（经典结果）
2. Approximate tilted distribution: $\hat{w}$ 使用 $\hat{A}$ 而非 $A^{\pi_k}$ → 引入 $\epsilon_A$ 
3. Flow matching approximation: $\hat{\pi}_{k+1}$ 与 $\pi_{k+1}^*$ 的 $W_2$ 距离 → performance difference lemma

### 4.2 Theorem 2: Multi-Modality Preservation

**定理2**（Flow PI保持多模态，Gaussian PI不保持）.
设 $\pi^*(a|s)$ 是 $M$-modal分布（$M \geq 2$），每个mode的mass $\geq \delta$。

**(a)** Gaussian PI的投影误差：

$$\text{KL}(\pi^* \| \hat{\pi}_{\text{Gaussian}}) \geq \log M - \log 2$$

**(b)** Flow PI的transport误差（与mode数量无关）：

$$W_2(\pi^*, \hat{\pi}_{\text{FPI}}) \leq \epsilon_{\text{FM}}$$

**含义**：Gaussian PI的误差随mode数量对数增长（因为要把多个mode坍缩为一个Gaussian），而Flow PI的误差只取决于flow matching的拟合能力——多少个mode进去，多少个mode出来。

**为什么这在VLA中重要**：π₀之所以使用flow matching，正是因为manipulation任务需要multi-modal action分布（同一个"放下物体"指令可以放在左边或右边）。如果RL fine-tuning中丧失了多模态性，等于丧失了使用flow matching的根本理由。

### 4.3 Theorem 3: Convergence Rate

**定理3**（全局收敛）.
在bounded reward ($|r| \leq R_{\max}$), Lipschitz MDP假设下，经过 $K$ 轮FPI：

$$V^* - V^{\pi_K} \leq \underbrace{\frac{\gamma^K}{1-\gamma} R_{\max}}_{\text{PI contraction}} + \underbrace{\frac{2\gamma L_Q}{(1-\gamma)^2} \epsilon_{\text{FM}}}_{\text{flow matching error}} + \underbrace{\frac{C}{(1-\gamma)^2} \epsilon_A}_{\text{advantage estimation error}}$$

**与Gaussian PI对比**：第二项中，flow matching的 $\epsilon_{\text{FM}}$ 替代了Gaussian投影的 $\epsilon_{\text{Gaussian}}$。由于flow matching的universal approximation性质，$\epsilon_{\text{FM}}$ 可以随网络capacity增大而趋于0，而Gaussian有**不可消除的structural error**。第三项 $\epsilon_A$ 对所有方法相同（取决于value estimation quality），这也正式表明advantage estimation是收敛的共同瓶颈。

### 4.4 Theorem 4: 统一现有方法

**定理4**（RWFM, ORW-CFM, AWM, FPO都是FPI的特例）.

**(a)** RWFM（Pfrommer et al., 2025）= FPI的单轮迭代（$K=1$），使用 from-scratch flow，不做iterative improvement。

**(b)** ORW-CFM-W₂（ICLR 2025）= FPI + W₂正则化的连续时间极限。

**(c)** AWM（Xue et al., 2025）= FPI在offline setting下的单轮特例，advantage由dataset returns定义。

**(d)** FPO（Berkeley, 2025）≈ FPI的一阶Taylor展开：FPO用CFM loss变化率近似log-likelihood ratio，本质是在近似FPI的exact improvement step的梯度方向。

**含义**：FPI不是又一个flow+RL方法，而是一个统一框架——多个独立提出的方法都是其特例。

---

## 五、FPI vs 所有现有方法的系统对比

### 5.1 技术难题的消除

| 技术挑战 | 现有方法如何挣扎 | FPI如何解决 |
|---------|----------------|-----------|
| Log-likelihood不可解 | FPO: CFM loss proxy; π_RL: noise network; SAC Flow: reparameterize | **不需要likelihood。** 训练目标是weighted MSE |
| Backprop through ODE不稳定 | SAC Flow: ≡RNN→GRU; QAM: adjoint绕过; RFO: gradient clip | **不需要backprop through ODE。** Standard flow matching loss |
| Policy gradient方差大 | FPO: 多次MC; π_RL: Flow-SDE exploration | **不用policy gradient。** Importance weighting + regression |
| 多模态collapse | ORW-CFM: W₂正则; FlowRL: W₂约束 | **天然保持多模态**（Theorem 2），无需额外正则 |
| On-policy效率低 | 所有on-policy方法每批数据只能更新一次 | **支持多epoch**：regression形式 + trust region保证稳定 |

### 5.2 训练流程简洁性对比

**FPO (Berkeley) 的训练循环**：
```
1. Rollout → (s, a, r)
2. 计算advantage via GAE
3. 计算CFM loss L_CFM 作为log-likelihood proxy
4. 构造PPO-clip surrogate objective
5. 多次MC sample估计ratio
6. Clipped gradient update
```

**π_RL (Flow-Noise) 的训练循环**：
```
1. Rollout → (s, a, r)
2. 计算advantage via Progress+TD
3. Forward pass through noise network to get log-likelihood
4. Compute policy ratio π_new/π_old
5. PPO-clip surrogate with structure-aware credit assignment
6. Update policy + noise network + Q-ensemble
```

**FPI的训练循环**：
```
1. Rollout → (s, a, r)
2. 计算advantage via Progress+TD value network
3. 计算weight w = normalized exp(A/λ)
4. 预计算 (τ, ε, v_old) 作为trust region锚
5. Weighted flow matching + trust region:
   L = w · ||v_θ - target||² + β_kl · ||v_θ - v_old||²
6. Standard gradient descent (多epoch稳定)
```

FPI少了likelihood computation、PPO-clip、ratio estimation。核心改动：给flow matching loss乘一个advantage weight + 加trust region防止漂移。

### 5.3 全方法对比表

| 方法 | 需要likelihood? | 需要BPTT? | 保持multi-modal? | 收敛证明? | Off-policy? | Value Estimation |
|------|:-:|:-:|:-:|:-:|:-:|------|
| **FPI (Ours)** | ✗ | ✗ | ✓ (Thm 2) | ✓ (Thm 3) | ✓ | Progress+TD |
| FPO (Berkeley) | ≈ proxy | ✗ | Partial | ✗ | ✗ | GAE (dense reward only) |
| FPO-VLA | ≈ proxy | ✗ | Partial | ✗ | ✗ | Structure-aware |
| π_RL | ✓ (noise net) | ✗ | Partial | ✗ | ✗ | Progress+TD |
| ReinFlow | ✓ (noise net) | ✗ | Partial | ✗ | ✗ | Standard |
| SAC Flow | ✓ (noise aug) | ✓ (stabilized) | Partial | ✗ | ✓ | SAC critic |
| QAM | ✗ | ✗ | ✓ | Partial | ✓ | TD Q-learning |
| FlowRL | ✗ | ✗ | ✓ (W₂) | ✗ | ✓ | Q-max |
| DSRL | ✗ (black box) | ✗ | Limited | ✗ | ✓ | SAC critic |
| RWFM | ✗ | ✗ | ✓ | ✗ | ✗ | Reward surrogate |
| ORW-CFM-W₂ | ✗ | ✗ | ✓ (W₂) | Partial | ✗ | Reward model |

### 5.4 Robustness Argument

当advantage estimation不准时（sparse reward早期训练的常态）：

**π_RL/FPO**：noisy advantage → noisy policy gradient → 更新方向可能错误 → 需要PPO-clip限制步长 → 保守更新 → 收敛慢

**FPI**：noisy advantage → noisy weight → weighted regression → **worst case回退为uniform BC**。关键区别：FPI永远不会"推向错误方向"——它只是在好的action上多学一些、差的action上少学一些。而policy gradient中负advantage会主动push away某些action，如果A估错了就会push away好的action。

**实践注意**：上述鲁棒性分析假设单步更新。多epoch更新时，即使w≈1（BC），velocity field也会因为过拟合rollout数据而漂移。因此实践中必须加trust region：$\beta_{\text{kl}} \cdot \|v_\theta - v_{\text{old}}\|^2$，保证多epoch训练稳定性。

---

## 六、实验方案

### 6.1 Research Questions

| RQ | 问题 | 验证方式 |
|----|------|---------|
| RQ1 | FPI是否match或超越现有flow+RL方法？ | LIBERO/ManiSkill success rate |
| RQ2 | FPI是否更好地保持multi-modal action distribution？ | Mode coverage定量测量 |
| RQ3 | FPI的training是否更稳定、更快？ | Wall-clock time, loss curves |
| RQ4 | Iterative PI（多轮）是否优于single-round RWFM？ | 消融：FPI-1轮 vs FPI-多轮 |
| RQ5 | FPI在π₀-scale (3B参数) 上是否可行？ | π₀ + LoRA fine-tuning |

### 6.2 Benchmarks

| Benchmark | Tasks | Reward | 选择理由 |
|-----------|-------|--------|---------|
| LIBERO-Spatial | 10 tasks | Binary success | 所有competitor的标准测试 |
| LIBERO-Object | 10 tasks | Binary success | Object generalization |
| LIBERO-Goal | 10 tasks | Binary success | Goal generalization |
| LIBERO-Long | 10 tasks | Binary success | Long-horizon (最难) |
| ManiSkill (多任务) | 4000+ pick-and-place | Binary success | 大规模并行验证 |
| ALOHA Sim (双臂) | 双臂操作 | Binary success | Multi-modality最重要 |

### 6.3 Baselines（直接competitor）

| 方法 | 来源 | 为什么必须对比 |
|------|------|---------------|
| π_RL (Flow-Noise) | Chen et al. 2025 | 直接在π₀上做RL，同样用Progress+TD value |
| π_RL (Flow-SDE) | Chen et al. 2025 | 另一种π₀ RL方案 |
| FPO-VLA | Lyu et al. 2025 | 直接在π₀上做FPO |
| ReinFlow | Zhang et al. 2025 | NeurIPS 2025 accepted |
| DSRL | Wagenmaker et al. 2025 | CoRL 2025, latent steering |
| SFT (BC baseline) | - | 不做RL的baseline |

### 6.4 Ablation Studies

| 消融 | 对比 | 验证什么 |
|------|------|---------|
| **FPI-K轮 vs FPI-1轮** | K=1 (≈RWFM) vs K=5 vs K=10 | Iterative improvement的价值 |
| **Temperature λ** | 固定 vs annealing schedule | Exploitation-exploration tradeoff |
| **Weight clipping** | $w_{\max}$ = 5, 10, 20, ∞ | Weight normalization的必要性 |
| **Update epochs** | 1 epoch vs 5 vs 10 per iteration | Off-policy data reuse |
| **Flow vs Gaussian PI** | FPI vs Gaussian advantage-weighted regression | Flow matching的必要性 |
| **Value: Prog+TD vs TD-only** | 有/无progress warm-start | Progress prior的贡献 |

### 6.5 Multi-Modality Analysis（独特卖点）

设计multi-solution LIBERO任务：如"将杯子放到盘子上"（可放左边/右边/中间）。

**Metric: Mode Coverage Rate**
1. 对每个state，从policy采样1000个action chunks
2. 用k-means聚类（k从1到10扫）
3. 用silhouette score确定最佳k
4. 统计discovered modes / ground-truth modes

**预期**：FPI保持2-3个modes，π_RL/FPO在RL fine-tuning后collapse到1个mode。

### 6.6 π₀ Scale Experiment

**Setup**:
- Model: π₀ (3B parameters) DROID weights
- Fine-tuning: LoRA rank=16 on action head
- Environments: 320 parallel (via RLinf framework)
- Benchmarks: LIBERO-Long + ManiSkill subset

**与π_RL的公平对比**:
- 相同compute budget（GPU-hours）
- 相同rollout数量
- 相同value network architecture (Progress+TD)
- **唯一差异**：policy improvement step（FPI vs Flow-Noise/Flow-SDE）

---

## 七、应对Reviewer质疑

### Q1: "这不就是reward-weighted regression / RWFM吗？"

**回应**：

RWFM是FPI的K=1特例（Theorem 4a明确指出）。FPI的核心贡献在于：

1. **Iterative framework + convergence theory** (Thm 1, 3)：RWFM没有PI的迭代结构和收敛分析
2. **Multi-modality guarantee** (Thm 2)：RWFM没有
3. **统一视角** (Thm 4)：FPI将RWFM/ORW-CFM/AWM/FPO识别为同一框架特例
4. **实验验证iterative > single-round**：消融实验直接展示K=5显著优于K=1

### Q2: "Advantage estimation用的是π_RL的方案，新意在哪？"

**回应**：

Value estimation不是本文的contribution——这是明确stated的。本文的contribution是**policy improvement mechanism**。正如PPO的contribution是clipping trick而不是GAE，FPI的contribution是"PI=distribution transport"这个insight + weighted flow matching作为improvement operator + 四个理论保证。

使用π_RL的value方案恰好**加强了公平对比**：在同样的advantage estimation下，纯粹比较policy improvement step。

### Q3: "Importance sampling在高维空间方差很大"

**回应**：

1. FPI不做importance sampling来估计期望——而是用importance weight做loss weighting。两者是不同的操作。Loss weighting的方差远小于IS期望估计。
2. Self-normalized weights + clipping进一步控制方差
3. 消融实验中展示不同 $w_{\max}$ 的影响
4. 理论上，Theorem 1的bound已经包含了approximation error

### Q4: "ORW-CFM-W₂证明了不加W₂正则化必然collapse（Lemma 1），FPI怎么避免？"

**回应**：

ORW-CFM的Lemma 1证明：在 **无限迭代 + 无约束** 情况下，online reward-weighted会collapse到Dirac。但FPI通过三个机制避免：

1. **温度 $\lambda > 0$**：exponential tilting的temperature天然提供entropy regularization
2. **有限迭代K**：实践中5-10轮即停
3. **Flow matching本身的inductive bias**：velocity field是smooth function，很难collapse到Dirac（需要无穷大gradient）

实验中可以直接测量policy entropy随PI轮数的变化来验证不collapse。

### Q5: "与QAM（2026.01）的关系？"

**回应**：

QAM = Adjoint Matching + TD Q-learning，用于off-policy RL。

关键区别：
- QAM需要**learned Q-function**提供action gradient → 额外的critic网络 + 训练instability
- FPI只需要**scalar advantage**做weight → 更简单
- QAM的adjoint matching需要backward ODE → 额外计算开销
- FPI的weighted flow matching就是forward MSE → 无额外开销
- QAM在offline/offline-to-online上测试; FPI面向online VLA fine-tuning

两者可互补：QAM可视为FPI在off-policy TD-learning setting下的一个更精细（但更复杂）的变体。

---

## 八、论文结构

**Title**: *Flow Policy Iteration: Reinforcement Learning as Distribution Transport*

**Subtitle (optional)**: *Unifying Reward-Weighted Flow Matching Methods with Policy Iteration Theory*

### Structure (9 pages + appendix)

**1. Introduction** (1.5 pages)
- 开头：所有flow+RL方法的共同矛盾
- 核心insight：PI = distribution transport = flow matching
- Key message + contributions列表

**2. Preliminaries** (1 page)
- Flow matching basics
- KL-regularized policy iteration
- π₀ architecture overview

**3. Flow Policy Iteration** (2 pages)
- 3.1 Insight: PI as transport
- 3.2 Value estimation (Progress + TD, credited to π_RL)
- 3.3 Weighted flow matching as improvement operator
- 3.4 Algorithm 1
- 3.5 Implementation details (LoRA, temperature schedule, weight clipping)

**4. Theoretical Analysis** (2 pages)
- Theorem 1: Improvement guarantee
- Theorem 2: Multi-modality preservation
- Theorem 3: Convergence rate
- Theorem 4: Unification of existing methods

**5. Experiments** (2.5 pages)
- 5.1 Main results (LIBERO, ManiSkill, ALOHA)
- 5.2 Multi-modality analysis
- 5.3 Ablations
- 5.4 π₀ scaling experiment
- 5.5 Training efficiency comparison

**6. Related Work** (0.5 pages)
- Concise taxonomy of flow+RL methods

**7. Conclusion** (0.5 pages)

**Appendix**:
- A: Full proofs of all theorems
- B: Extended experimental results
- C: Hyperparameter sensitivity
- D: Detailed comparison with RWFM/ORW-CFM/AWM

---

## 九、时间线

| 时间 | 里程碑 |
|------|--------|
| 2026.03 W1-2 | Theorem 1-4 完整证明 |
| 2026.03 W3-4 | LIBERO prototype（验证FPI > BC, FPI-K轮 > FPI-1轮） |
| 2026.04 | 完整LIBERO实验 + baseline对比 |
| 2026.05 | ManiSkill + ALOHA + multi-modality分析 |
| 2026.06 | π₀ scaling experiment (LoRA + RLinf) |
| 2026.07 W1-2 | 论文写作 |
| 2026.07 W3 | 内部review + 修改 |
| 2026.07底 | 提交NeurIPS 2026 |

---

## 十、风险评估

| 风险 | 等级 | Mitigation |
|------|------|-----------|
| Reviewer说"就是RWFM" | **高** | Theorem 4 + ablation(K=1 vs K=5) + multi-modality experiment |
| 实验不够显著 | 中 | 选multi-modal任务（FPI优势最大）; 公平对比（同value，同compute） |
| 某组抢发类似idea | 中 | 尽早提交; 理论深度是壁垒（4个定理不容易复制） |
| π₀实验资源不足 | 中 | LoRA降低compute; 先在小模型验证; 必要时只做LIBERO |
| Temperature λ敏感 | 低 | Annealing schedule + ablation展示robustness |

---

## 十一、一句话总结

> **Policy improvement本质是distribution transport。Flow matching天然做distribution transport。所以flow matching不需要嫁接外部RL算法——它本身就是RL算法。给flow matching loss乘一个advantage weight，就完成了理论上optimal的policy improvement step。**
