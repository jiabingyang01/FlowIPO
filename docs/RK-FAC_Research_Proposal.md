# Residual-Kinetic Flow Actor-Critic (RK-FAC)：基于残差动能信任域的 Flow VLA Off-Policy RL

## NeurIPS 2026 完整研究方案

---

## 一、Motivation：一个被忽略的结构性矛盾

### 1.1 现有范式的共同困境

截至 2026 年 3 月，Flow VLA + RL 领域已有 30+ 篇论文。它们在三个维度上形成了一个**不可能三角**：

| | Temporal Credit (Long-horizon) | OOD Generalization | Sample Efficiency |
|---|:-:|:-:|:-:|
| πRL (PPO+Critic) | ✓ TD bootstrap | ✗ critic 过拟合视觉纹理 | ✗ on-policy |
| pi-StepNFT (no critic) | ✗ episode binary | ✓ 纯环境 ground truth | ✗ on-policy |
| FPO++ (CFM proxy) | ✓ GAE | ? 未测 OOD | ✗ on-policy |
| DSRL (noise SAC) | ✓ TD in noise space | ✓ 不改权重 | ✓ off-policy |
| FPI (advantage-weighted FM) | ✓ Progress+TD | ? 未测 OOD | Partial (多epoch) |
| RECAP/RAMP | Partial | ✓ offline | ✗ 离线 |

**没有任何方法同时解决三个问题。**

ManiSkill OOD 数据揭示了一个被严重低估的现象：

- pi-StepNFT Semantic OOD: **49.1%** vs PPO: **25.4%** — 几乎翻倍
- pi-StepNFT Vision OOD: **69.1%** vs PPO: **61.1%**
- pi-StepNFT Execution OOD: **33.1%** vs PPO: **31.5%**

**这意味着 PPO 的 critic 在 OOD 场景下是一个严重的 liability，不是 asset。**

### 1.2 根本原因分析

PPO 的 critic V(s) 从 VLM 的 visual-language embedding 端到端学习价值函数。训练过程中：

1. Critic 的梯度回传到 VLM backbone → 视觉特征被 reward signal 污染
2. Critic 学会了将训练分布中的**视觉纹理**（特定物体颜色、桌面纹理、光照）与 reward 关联
3. 在 OOD 场景中（新物体、新背景），这些纹理关联失效 → critic 给出错误的价值估计 → 策略被误导

pi-StepNFT 的 OOD 优势**不是因为没有 critic**，而是因为**没有从视觉特征学到的 spurious correlation**。

### 1.3 核心 Insight

**FLAC (2026.02) 证明了一件关键的事：flow 策略的 MaxEnt RL 不需要计算似然——用速度场的动能（kinetic energy）就够了。**

$$E(s) = \mathbb{E}\left[\int_0^1 \frac{1}{2}\|u_\theta(s, \tau, X_\tau)\|^2 d\tau\right]$$

通过 Girsanov 定理：$D_{\text{KL}}(P_\theta \| P_{\text{ref}}) = \frac{1}{\sigma^2} E(s)$

但 FLAC 只在 DMControl 小任务上做了 from-scratch 训练。**VLA fine-tuning 的参考过程不是 Brownian motion（零 drift），而是预训练策略（非零 drift）。**

**RK-FAC 的核心创新**：定义**残差动能**——当前策略相对于预训练策略的路径空间偏移量——作为 flow VLA RL 的 likelihood-free 信任域，同时提供 OOD 自动保护机制。

**一句话 message**：*Flow VLA fine-tuning 的信任域不应该在似然比空间（PPO clipping）或速度场距离（ad-hoc L2）中定义，而应该在路径空间的物理量——残差动能——中定义。这给出了精确的 KL bound，自动在 OOD 状态下回退到预训练行为。*

---

## 二、相关工作全景（截至 2026.03）

### 2.1 Flow VLA + RL：按似然处理方式分类

| 路线 | 代表 | 似然处理 | Critic | OOD | 问题 |
|------|------|---------|:------:|:---:|------|
| 近似似然 + PPO | πRL, ReinFlow | SDE noise network | ✓ V(s) | 差 | Critic 过拟合视觉纹理 |
| CFM proxy ratio | FPO++ | CFM loss差值 | ✓ V(s) | 未测 | Proxy 不精确 |
| Contrastive mirror | pi-StepNFT | 完全绕过 | ✗ | **好** | Long-horizon 信用分配差 |
| Advantage-weighted FM | FPI, RWFM, AWM | 完全绕过 | ✓ V(s) | 未测 | Importance weight 方差 |
| Noise-space RL | DSRL | N/A（不改权重） | ✓ Q(s,ε) | 好 | 表达力天花板 |
| 动能正则化 | FLAC | 完全绕过 | ✓ Q(s,a) | 未测 | 仅 from-scratch，未用于 VLA |

### 2.2 OOD Robustness 的已有认知

| 论文 | OOD 相关发现 |
|------|-------------|
| pi-StepNFT | Critic-free → OOD +11%。归因：不训额外模型，只用环境 ground truth |
| DSRL | 不改 VLA 权重 → OOD 天然保持。归因：VLA pretrained features 未被破坏 |
| PLD | 残差 RL expert + 蒸馏回 VLA → OOD 保持。归因：VLA 权重通过 BC 更新 |
| LRM | Binary rcomp 在闭环 RL 中最好（60.93%）。归因："涌现同步" |

**共同规律**：OOD robustness 与"VLA pretrained features 被破坏的程度"负相关。任何从 visual features 端到端学习的组件（critic、reward model）都是 OOD 的潜在风险源。

### 2.3 FLAC 的 GSB 框架

FLAC 将 MaxEnt RL 建模为 Generalized Schrödinger Bridge 问题：

$$\min_P J_{\text{GSB}}(P) = \alpha \cdot D(P \| P_{\text{ref}}) + \mathbb{E}_{X_1 \sim P}[G(X_1)]$$

核心结果：
- 路径散度 $D_{\text{KL}}(P_\theta \| P_{\text{ref}}) = \frac{1}{\sigma^2} E(s)$（Girsanov）
- 终端散度被路径散度上界（DPI）
- 动能在 ODE 求解时自然计算，零额外开销
- NFE=2 即可达到 DIME (NFE=16) 的性能

**FLAC 的局限**：参考过程 $P_{\text{ref}}$ 是 Brownian motion（零 drift）。VLA fine-tuning 需要参考 pretrained policy（非零 drift）。

### 2.4 定位总结

**RK-FAC = FLAC 的 GSB 理论框架 + 残差动能信任域 + frozen-VLM Q-network + VLA fine-tuning**

它不是对任何现有方法的增量修改，而是把一个 principled 的 RL 理论框架适配到 VLA fine-tuning 的特定需求上。

---

## 三、方法：Residual-Kinetic Flow Actor-Critic (RK-FAC)

### 3.1 Problem Setup

**环境**：MDP $(S, A, P, R, \gamma)$，$A \subseteq \mathbb{R}^{d_a}$ 连续动作空间。

**初始策略**：$\pi_{\text{pre}}$ = pre-trained flow matching VLA（如 π₀/π₀.₅），velocity field $u_{\text{pre}}(x_\tau, \tau \mid o, \ell)$。

**目标**：Fine-tune 得到 $\pi_\theta$，最大化任务成功率，同时保持 OOD 泛化能力。

**Architecture**：
- Frozen VLM backbone（PaliGemma-3B）
- Trainable flow action expert（~300M 参数，和 pi-StepNFT 相同）
- Q network：$Q_\phi(\text{sg}[h_{\text{VLM}}(o)], a)$，~10M 参数，两个 Q head
- 只更新 action expert 和 Q network，VLM backbone 完全冻结

### 3.2 残差动能：从 FLAC 到 VLA Fine-Tuning

#### 3.2.1 定义

当前策略 $\pi_\theta$ 的 flow action expert 产生速度场 $u_\theta(x_\tau, \tau \mid o, \ell)$。预训练策略产生 $u_{\text{pre}}(x_\tau, \tau \mid o, \ell)$（frozen，一次 forward pass）。

**残差动能**：

$$E_{\text{res}}(s) = \mathbb{E}_{X_\tau \sim P_\theta}\left[\int_0^1 \frac{1}{2}\|u_\theta(s, \tau, X_\tau) - u_{\text{pre}}(s, \tau, X_\tau)\|^2 d\tau\right]$$

物理直觉：残差动能衡量当前策略在生成动作时，速度场偏离预训练策略的"做功量"。动能越大 = 偏离越远 = 越不安全。

#### 3.2.2 理论保证

**Theorem 1 (Residual KL Bound)**. 设 $P_\theta$ 和 $P_{\text{pre}}$ 分别是当前策略和预训练策略的路径测度，两者共享相同的 SDE 扩散系数 $\sigma > 0$。则：

$$D_{\text{KL}}(P_\theta \| P_{\text{pre}}) = \frac{1}{\sigma^2} E_{\text{res}}(s)$$

由数据处理不等式（DPI），终端动作分布满足：

$$D_{\text{KL}}(\pi_\theta(\cdot|s) \| \pi_{\text{pre}}(\cdot|s)) \leq \frac{1}{\sigma^2} E_{\text{res}}(s)$$

**证明思路**：
1. 两个 SDE 共享相同的扩散系数 $\sigma$，drift 分别为 $u_\theta$ 和 $u_{\text{pre}}$
2. 由 Girsanov 定理，$\frac{dP_\theta}{dP_{\text{pre}}} = \exp\left(\frac{1}{\sigma^2}\int_0^1 (u_\theta - u_{\text{pre}})^T dW_\tau - \frac{1}{2\sigma^2}\int_0^1 \|u_\theta - u_{\text{pre}}\|^2 d\tau\right)$
3. 取 KL 散度期望，Itô isometry 消去交叉项
4. 余项恰为残差动能除以 $\sigma^2$

**含义**：最小化残差动能 = 在路径空间约束策略不偏离预训练策略。这比 PPO 的 clipping（在近似似然比空间操作）和 pi-StepNFT 的 $\lambda\|\Delta v\|^2$（ad-hoc L2 惩罚）都更 principled。

#### 3.2.3 OOD 自动保护机制

**Theorem 2 (OOD Fallback)**. 设 $\pi_\theta$ 由 RK-FAC 训练得到，$\alpha > 0$ 为残差动能的正则系数。对任意状态 $s$（包括 OOD 状态）：

$$\|\pi_\theta(\cdot|s) - \pi_{\text{pre}}(\cdot|s)\|_{\text{TV}} \leq \sqrt{\frac{E_{\text{res}}(s)}{2\sigma^2}}$$

**含义**：在 OOD 状态上，如果 Q 估计不可靠，actor 更新会试图大幅偏离预训练行为 → 残差动能急剧增大 → 被 $\alpha$ 惩罚压回 → 策略自动回退到预训练行为。

**与 PPO 的对比**：PPO 的 KL 约束在似然比空间操作。对 flow 策略，似然不可计算，PPO 只能用近似值——近似的 KL 约束在 OOD 状态上可能失效。RK-FAC 的残差动能是**精确可计算的**（ODE 求解时同步计算），不依赖任何近似。

#### 3.2.4 $E_{\text{res}}$ 的计算

在 K 步 Euler ODE 求解时同时计算残差动能：

```
E_res = 0
for k in range(K):
    tau_k = k / K
    u_theta_k = action_expert_theta(x_k, tau_k, context)  # 当前策略
    u_pre_k = action_expert_pre(x_k, tau_k, context)       # frozen 预训练（一次 forward）
    E_res += 0.5 * ||u_theta_k - u_pre_k||^2 * (1/K)       # 梯形积分
    x_{k+1} = x_k + u_theta_k * (1/K)                       # Euler step
a = x_K  # 最终动作
```

额外成本 = 一次 frozen action expert forward per ODE step。K=4 时，额外 4 次 ~300M 模型 forward — 对 VLA 的总推理成本（VLM forward 是瓶颈）可忽略。

### 3.3 Q-Network 设计：OOD-Robust Critic

#### 3.3.1 Frozen-VLM Q

$$Q_\phi(s, a) = \text{MLP}_\phi\bigl(\text{sg}[h_{\text{VLM}}(o, \ell)], \; a\bigr)$$

- $\text{sg}[\cdot]$: stop-gradient。Q 的梯度**不回传到 VLM backbone**
- $h_{\text{VLM}}$: frozen VLM 的最后一层 hidden state
- MLP: 2-3 层，~5-10M 参数

**为什么这比 PPO 的 critic 更 OOD-robust**：

PPO 的 critic 端到端训练 VLM → 视觉特征被 reward 污染 → OOD 崩溃。

RK-FAC 的 Q 只在 frozen VLM features 上训练 MLP → VLM 的 internet-scale pretrained features 不被破坏 → 对 OOD 物体/场景的表征偏移有限。

**Theorem 3 (Q Generalization Bound)**. 设 VLM 特征的 OOD 偏移为 $\delta_h = \|h_{\text{VLM}}(\mathcal{D}_{\text{OOD}}) - h_{\text{VLM}}(\mathcal{D}_{\text{IND}})\|$，MLP 的 Lipschitz 常数为 $L_{\text{MLP}}$。则：

$$|Q_\phi^{\text{OOD}}(s, a) - Q_\phi^{\text{IND}}(s, a)| \leq L_{\text{MLP}} \cdot \delta_h$$

对比 PPO 的 trainable-VLM critic：

$$|V_\psi^{\text{OOD}}(s) - V_\psi^{\text{IND}}(s)| \leq L_{\text{full}} \cdot (\delta_{\text{input}} + \delta_{\text{VLM-drift}})$$

其中 $\delta_{\text{VLM-drift}}$ 是训练导致的 VLM 特征漂移——RK-FAC 中这一项为零。

#### 3.3.2 双 Q + Target Network

标准 SAC 式设计：

- 两个 Q network $Q_{\phi_1}, Q_{\phi_2}$，取 min 减少过估计
- Target network $\bar{Q}$ 用 EMA 更新：$\bar{\phi} \leftarrow \eta \bar{\phi} + (1-\eta)\phi$

#### 3.3.3 Replay Buffer

维护 replay buffer $\mathcal{B}$，存储 $(o_t, \ell, a_t, r_t, o_{t+1})$。Off-policy 数据复用，无需每轮清空。

### 3.4 Policy Improvement via Q-Gradient + Residual Kinetic Trust Region

#### 3.4.1 Actor Loss

$$\mathcal{L}_{\text{actor}}(\theta) = \mathbb{E}_{s \sim \mathcal{B}} \left[ \alpha \cdot E_{\text{res}}(s) - Q_\phi(s, a_\theta(s)) \right]$$

其中 $a_\theta(s)$ 是 flow policy 从 $\epsilon \sim \mathcal{N}(0, I)$ 经 K 步 ODE 生成的动作。Q-gradient 通过可微 ODE solver 反传到 $\theta$。

**直觉**：
- $-Q_\phi(s, a)$ 项：推动策略生成高价值动作
- $\alpha \cdot E_{\text{res}}(s)$ 项：惩罚偏离预训练策略的行为
- $\alpha$ 自动调节两者的 tradeoff

#### 3.4.2 Critic Loss

Energy-regularized soft Bellman target：

$$y = r + \gamma \left[\min_{j=1,2} Q_{\bar{\phi}_j}(s', a') - \alpha \cdot E_{\text{res}}(s')\right]$$

$$\mathcal{L}_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\left[\frac{1}{2}(Q_\phi(s, a) - y)^2\right]$$

这和 FLAC 的 energy-regularized Bellman 算子结构相同，但用残差动能替代了绝对动能。

#### 3.4.3 Automatic α Tuning

设目标残差动能预算 $\mathcal{E}_{\text{tgt}}$，通过 Lagrangian 对偶自动调参：

$$\log\alpha \leftarrow \log\alpha - \beta_\alpha \cdot \mathbb{E}_{s \sim \mathcal{B}}[\mathcal{E}_{\text{tgt}} - \text{sg}(E_{\text{res}}(s))]$$

**几何启发式**：$\mathcal{E}_{\text{tgt}} = C \cdot d_a$，$C \in [0.1, 1.0]$，$d_a$ 是动作维度。

训练动态：
- 早期策略接近预训练 → $E_{\text{res}}$ 小 → $\alpha$ 下降 → 允许更大偏离 → 加速学习
- 后期策略改善 → 尝试更激进的偏离 → $E_{\text{res}}$ 上升 → $\alpha$ 上升 → 自动收紧 → 防止过拟合

### 3.5 完整算法

---

#### Algorithm 1: Residual-Kinetic Flow Actor-Critic (RK-FAC)

**Input:** Pre-trained flow VLA $\pi_{\text{pre}}$（frozen），可训练 action expert $u_\theta$（初始化为 $u_{\text{pre}}$），Q networks $Q_{\phi_1}, Q_{\phi_2}$，target networks $\bar{Q}_1, \bar{Q}_2$，replay buffer $\mathcal{B}$，温度 $\alpha$（可学习），目标动能 $\mathcal{E}_{\text{tgt}}$

**For** each training iteration **do:**

**Step 1 (Rollout & Store):** 用 $\pi_\theta$ 在 $N_{\text{env}}$ 个并行环境中执行，收集 transitions $(o_t, \ell, a_t, r_t, o_{t+1})$ 存入 $\mathcal{B}$

**Step 2 (Sample Batch):** 从 $\mathcal{B}$ 中采样 mini-batch $\{(o, \ell, a, r, o')\}$

**Step 3 (Q Update):** 对 mini-batch 中的 $(o', \ell)$，用当前策略生成 $a' = \text{ODE}(u_\theta, \epsilon'; K)$ 并计算 $E_{\text{res}}(s')$

$$y = r + \gamma\left[\min_{j} \bar{Q}_j(o', a') - \alpha \cdot E_{\text{res}}(s')\right]$$
$$\phi_j \leftarrow \phi_j - \alpha_Q \nabla_{\phi_j} \frac{1}{2}\|Q_{\phi_j}(o, a) - y\|^2, \quad j = 1, 2$$

**Step 4 (Actor Update):** 对 mini-batch 中的 $(o, \ell)$，重新生成 $a_\theta = \text{ODE}(u_\theta, \epsilon; K)$，同步计算 $E_{\text{res}}(s)$

$$\theta \leftarrow \theta - \alpha_\pi \nabla_\theta \left[\alpha \cdot E_{\text{res}}(s) - \min_j Q_{\phi_j}(o, a_\theta)\right]$$

**Step 5 (α Update):**

$$\log\alpha \leftarrow \log\alpha - \beta_\alpha(\mathcal{E}_{\text{tgt}} - \text{sg}(E_{\text{res}}(s)))$$

**Step 6 (Target Update):**

$$\bar{\phi}_j \leftarrow \eta \bar{\phi}_j + (1 - \eta)\phi_j, \quad j = 1, 2$$

**End For**

**Return** $\pi_\theta$

---

### 3.6 关键实现细节

#### 兼容 π₀/π₀.₅ 架构

π₀ 的结构是 VLM encoder → flow matching action head (Action Expert DiT)。RK-FAC：
- **冻结 VLM encoder**（完全不动）
- **训练 Action Expert DiT**（通过 Q-gradient + 残差动能约束）
- **训练 Q-MLP**（在 frozen VLM features 上）

与 LoRA 兼容：可以只对 Action Expert 的 attention 层加 LoRA，进一步降低可训练参数。

#### Q-Gradient 穿过 ODE 的稳定性

Q-gradient 需要反传穿过 K 步 ODE solver。K=4 时等价于 4 层的 chain rule。

**已有验证**：SAC Flow (2026) 已经在类似设置下验证了 ODE 反传的可行性（使用 GRU 重参数化）。RK-FAC 不需要 GRU——因为残差动能天然提供了正则化，限制了速度场的幅度，从而限制了梯度的幅度。

**额外稳定化**：gradient clipping on actor update（clip norm ≤ 1.0）。

#### Reward 设计

使用最简单的 binary reward：$r_T = +1$（成功），$r_T = -1$（失败），$r_t = 0$（中间步）。

Q 通过 TD bootstrap 自动将 terminal reward 传播到每一步。这就是 temporal credit assignment 的来源——不需要 GAE、不需要 progress prior、不需要 VLM reward model。

#### NFE 选择

训练和推理都用 **K=4**（和 pi-StepNFT 一致）。FLAC 的结论支持低 NFE：动能正则化偏好低能量（接近直线）的传输路径，少量步数即可准确求解。

---

## 四、理论分析

### 4.1 Theorem 1: Residual KL Bound（已在 3.2.2 给出）

残差动能精确等于路径空间 KL 散度（在 SDE regime 下），上界终端分布 KL（通过 DPI）。

### 4.2 Theorem 4: Policy Improvement Guarantee

**定理4**（Monotonic Improvement with Residual Kinetic Trust Region）.
设 $\pi_{k+1}$ 由 Algorithm 1 的一步 actor update 产生，Q estimation error 满足 $\|Q_\phi - Q^{\pi_k}\|_\infty \leq \epsilon_Q$。则：

$$V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s) - \frac{2\gamma}{(1-\gamma)^2}\left(L_a \cdot \sqrt{2\sigma^2 \cdot \bar{E}_{\text{res}}} + \epsilon_Q\right)$$

其中 $L_a$ 是 Q 关于 action 的 Lipschitz 常数，$\bar{E}_{\text{res}}$ 是平均残差动能。

**含义**：
1. 残差动能 $\bar{E}_{\text{res}}$ 越小（策略越接近预训练），improvement 越保守但越安全
2. $\alpha$ 的 automatic tuning 自动控制 $\bar{E}_{\text{res}}$ 的水平
3. Q estimation error $\epsilon_Q$ 显式出现在 bound 中——这激励 frozen-VLM Q 的设计（更小的 $\epsilon_Q$ 在 OOD 上）

### 4.3 Theorem 5: OOD Robustness Guarantee

**定理5**（OOD Performance Lower Bound）.
设 $\pi_\theta$ 由 RK-FAC 训练得到，训练分布为 $\mathcal{D}_{\text{IND}}$，测试分布为 $\mathcal{D}_{\text{OOD}}$。设 VLM 特征偏移 $\delta_h = \sup_s \|h_{\text{VLM}}(s_{\text{OOD}}) - h_{\text{VLM}}(s_{\text{IND}})\|$，MLP Q 的 Lipschitz 常数为 $L_Q$。则：

$$V^{\pi_\theta}_{\text{OOD}} \geq V^{\pi_{\text{pre}}}_{\text{OOD}} - \frac{L_a \sqrt{2\sigma^2 \bar{E}_{\text{res}}}}{1 - \gamma} - \frac{L_Q \cdot \delta_h}{(1-\gamma)^2}$$

**含义**：
- 第一项 $V^{\pi_{\text{pre}}}_{\text{OOD}}$：预训练策略在 OOD 上的性能（VLA pretrain 的质量）
- 第二项：残差动能引起的偏离损失（被 $\alpha$ 控制）
- 第三项：Q 在 OOD 上的估计误差传播（被 frozen VLM 限制）

**对比 PPO**：PPO 没有第二项的等价物（clipping 在 OOD 上不提供约束），且第三项中 $\delta_h$ 被 VLM 训练导致的特征漂移放大。

### 4.4 Theorem 6: 与 FLAC、SAC、pi-StepNFT 的关系

**定理6**（统一视角）.

**(a)** FLAC = RK-FAC 的特例，令 $u_{\text{pre}} = 0$（Brownian motion 参考），from-scratch 训练。

**(b)** SAC（高斯策略）= RK-FAC 的退化情形，令 K=1（单步 ODE = 仿射变换），动能退化为 $\frac{1}{2}\|\mu_\theta - \mu_{\text{pre}}\|^2$（均值偏移的 L2 范数）。

**(c)** pi-StepNFT 的信任域 $\lambda\|\Delta v_\theta\|^2$ ≈ RK-FAC 的残差动能在 $\tau = t$ 处的单点近似（不积分整条路径，不除以 $\sigma^2$，不与 KL bound 关联）。

---

## 五、RK-FAC vs 所有现有方法的系统对比

### 5.1 技术难题的解决方式

| 技术挑战 | 现有方法 | RK-FAC |
|---------|---------|--------|
| Log-likelihood 不可解 | FPO: proxy; πRL: noise net; SAC Flow: reparameterize | **不需要。** 动能 = likelihood-free KL proxy |
| Backprop through ODE | SAC Flow: GRU; QAM: adjoint | **直接反传，** 残差动能正则化限制梯度幅度 |
| OOD critic overfitting | PPO: 端到端 VLM → critic 过拟合 | **Frozen-VLM Q：** 特征不被破坏 |
| Temporal credit assignment | pi-StepNFT: episode binary → Long 差 | **Q via TD bootstrap：** step-level 信号 |
| 信任域定义 | PPO: clipping on approx ratio; StepNFT: ad-hoc L2 | **残差动能 = 精确 KL bound** |
| Sample efficiency | 所有 on-policy 方法 | **Off-policy replay buffer** |

### 5.2 全方法对比表

| 方法 | 似然？ | Critic | 信任域 | Off-policy | OOD 机制 | Long-horizon |
|------|:------:|:------:|--------|:----------:|---------|:------------:|
| πRL (PPO) | ≈ | V(s), trainable VLM | Clipping | ✗ | 无 | ★★★★★ |
| pi-StepNFT | ✗ | 无 | $\lambda\|\Delta v\|^2$ | ✗ | 无 critic | ★★★ |
| FPO++ | ≈ | V(s) | ASPO clip | ✗ | 未知 | ★★★★ |
| FPI | ✗ | V(s) | Weight clip + v_old L2 | Partial | 未知 | ★★★★ |
| DSRL | N/A | Q(s,ε), small | 不改权重 | ✓ | 不改权重 | ★★★ |
| SAC Flow | ≈ | Q(s,a) | SAC entropy | ✓ | 未知 | ★★★★ |
| FLAC | ✗ | Q(s,a) | 绝对动能 | ✓ | 未知 | ★★★★ |
| **RK-FAC** | **✗** | **Q(s,a), frozen VLM** | **残差动能 (KL bound)** | **✓** | **理论保证 (Thm 5)** | **★★★★★** |

### 5.3 与 FPI 的关系

FPI 和 RK-FAC 解决的是同一个大问题（flow VLA RL），但路线完全不同：

| 维度 | FPI | RK-FAC |
|------|-----|--------|
| Policy improvement | Advantage-weighted regression | Q-gradient through ODE |
| 信任域 | $\|v_\theta - v_{\text{old}}\|^2$（ad-hoc L2） | 残差动能（KL bound） |
| Value estimation | V(s) + Progress + TD | Q(s,a) + TD only |
| 理论基础 | KL-regularized PI = transport | GSB = path-space optimization |
| Off-policy | 多 epoch（需 trust region 稳定） | 真 off-policy（replay buffer） |
| OOD 分析 | 未做 | 有理论保证（Thm 5） |
| 多模态保持 | 有理论保证（FPI Thm 2） | 通过 $\alpha$ 控制（动能小 = 接近预训练 = 多模态保持） |

**可以互补**：RK-FAC 的 frozen-VLM Q 设计可以被 FPI 采用（替换其 V(s) 的 trainable critic 为 frozen-VLM critic）。RK-FAC 的 OOD 理论可以为 FPI 提供 robustness 分析。

---

## 六、实验方案

### 6.1 Research Questions

| RQ | 问题 | 验证方式 |
|----|------|---------|
| RQ1 | RK-FAC 能否同时在 Long-horizon 和 OOD 上达到 SOTA？ | LIBERO-Long + ManiSkill OOD |
| RQ2 | 残差动能信任域是否优于 PPO clipping 和 ad-hoc L2？ | 消融：残差 KE vs abs KE vs L2 vs clipping |
| RQ3 | Frozen-VLM Q 是否比 trainable-VLM V 更 OOD-robust？ | 消融：frozen vs trainable critic input |
| RQ4 | Off-policy reuse 是否显著提升 sample efficiency？ | Wall-clock time & rollout 数量对比 |
| RQ5 | Automatic α tuning 是否有效控制 exploration-exploitation？ | α 曲线 + 消融 auto vs fixed |

### 6.2 Benchmarks

| Benchmark | Tasks | 选择理由 |
|-----------|-------|---------|
| LIBERO-Spatial | 10 tasks | 短时域基准 |
| LIBERO-Object | 10 tasks | Object 泛化 |
| LIBERO-Goal | 10 tasks | Goal 泛化 |
| LIBERO-Long | 10 tasks | **Long-horizon（核心验证点）** |
| ManiSkill IND | 4352 pick-and-place | 大规模 IND |
| ManiSkill OOD-Vision | 未见纹理/光照 | **OOD robustness（核心验证点）** |
| ManiSkill OOD-Semantic | 未见物体/指令 | **OOD robustness（核心验证点）** |
| ManiSkill OOD-Execution | 未见初始构型 | **OOD robustness** |

### 6.3 Baselines

| 方法 | 为什么必须对比 |
|------|---------------|
| πRL (PPO, Flow-SDE) | 直接竞争者：有 temporal credit，OOD 差 |
| pi-StepNFT | 直接竞争者：OOD 好，Long 差 |
| FPI (Advantage-weighted FM) | 同为新方法，路线不同 |
| DSRL | Off-policy + 不改权重 |
| SFT (BC baseline) | 不做 RL 的 baseline |

### 6.4 Ablation Studies

| 消融 | 对比 | 验证什么 |
|------|------|---------|
| **残差 KE vs 绝对 KE vs L2 vs PPO clip** | 4 种信任域 | 残差动能的必要性 |
| **Frozen-VLM Q vs Trainable-VLM V** | Critic 架构 | OOD robustness 的来源 |
| **Auto α vs Fixed α** | α 策略 | Automatic tuning 的价值 |
| **K=2 vs K=4 vs K=8** | ODE 步数 | NFE 对性能的影响 |
| **Replay buffer size** | 1k, 10k, 50k | Off-policy reuse 的边际收益 |
| **Q-gradient vs Advantage-weighted (FPI-style)** | Policy improvement 方式 | 两种路线的对比 |

### 6.5 核心预期结果

| | Spatial/Object/Goal | **Long** | **OOD Avg** |
|---|:-:|:-:|:-:|
| πRL (PPO) | ~98% | **93%** | 39.3% |
| pi-StepNFT | ~95% | 80-87% | **50.4%** |
| FPI | ~97% | ~91% | ? |
| **RK-FAC** | ~97% | **91-94%** | **≥48%** |

**核心 claim**：RK-FAC 是第一个同时在 Long-horizon 和 OOD 上 match 或 beat PPO 和 pi-StepNFT 各自最强维度的方法。

---

## 七、应对 Reviewer 质疑

### Q1: "这不就是 FLAC 用到 VLA 上吗？"

**回应**：

1. **FLAC 用绝对动能（相对 Brownian motion），RK-FAC 用残差动能（相对预训练策略）。** 这不是简单的参数替换——需要重新推导 Girsanov 定理在非零 drift 参考下的形式（Theorem 1），以及 OOD fallback guarantee（Theorem 2/5，FLAC 没有）。

2. **FLAC 做 from-scratch 训练，RK-FAC 做 fine-tuning。** 这是完全不同的问题设定。Fine-tuning 的核心挑战是"如何改进策略的同时不破坏预训练的 OOD 泛化能力"——FLAC 完全没有 address 这个问题。

3. **Frozen-VLM Q 的设计**是针对 VLA OOD 问题的 specific contribution，不存在于 FLAC 中。

4. **实验维度不同**：FLAC 在 DMControl（小规模连续控制），RK-FAC 在 LIBERO/ManiSkill（VLA 规模 + OOD 验证）。

### Q2: "Q-gradient through ODE 稳不稳定？"

**回应**：

1. SAC Flow (2026) 已验证类似路径可行（GRU 重参数化速度网络 + off-policy Q-learning）
2. RK-FAC 比 SAC Flow 更稳定：残差动能天然限制速度场幅度 → 限制梯度幅度
3. K=4 很小（不是 50 步 diffusion），chain rule 深度有限
4. 消融实验展示 K=2/4/8 的稳定性

### Q3: "Frozen-VLM Q 的 capacity 够不够？"

**回应**：

1. Q 只需要区分"好动作 vs 差动作"在同一状态下的差异，不需要精确估计绝对值
2. VLM hidden state 已经编码了丰富的视觉-语言信息（这就是 VLM pretrain 的价值）
3. 实验消融中对比 MLP-2层 vs MLP-4层 vs Transformer-1层
4. 如果 frozen Q 不够 expressive → $\epsilon_Q$ 大 → Theorem 4 的 bound 松 → 但残差动能的保护仍然有效

### Q4: "和 FPI 比有什么优势？"

**回应**：

1. **OOD 理论保证**（Theorem 5）：FPI 没有 OOD 分析
2. **Off-policy**：RK-FAC 用 replay buffer 实现真正的 off-policy，FPI 的多 epoch 只是 near-on-policy
3. **信任域更 principled**：残差动能 = 精确 KL bound（Theorem 1），FPI 的 $\|v_\theta - v_{\text{old}}\|^2$ 是 ad-hoc L2

但也有**劣势**：
- 需要 backprop through ODE（FPI 不需要）
- 需要训练 Q network（FPI 只需要 V network）
- 实现复杂度更高

两者可互补，在同一篇论文中可以做 head-to-head 对比。

### Q5: "pi-StepNFT 的消融显示 learned signal 不如 binary，Q-based temporal credit 真的能 work 吗？"

**回应**：

这是最需要诚实面对的问题。pi-StepNFT 的消融确实显示 GAE/GRPO advantage 不如 binary。但有两个关键区别：

1. **pi-StepNFT 的消融是在 contrastive mirror loss 框架内做的**——把 binary y 换成 continuous advantage。在这个框架中，advantage 的噪声被 softplus 放大。RK-FAC 的 Q 不参与 loss 的 sign（不是 y），而是参与 gradient direction（Q-gradient），这是不同的信息利用方式。

2. **pi-StepNFT 的 advantage 来自 GRPO（组内归一化）或 GAE（需要 trainable V）。** RK-FAC 的 Q 是 off-policy TD-learned + frozen-VLM features——信息来源和训练方式都不同。

**但这仍然是一个风险**。如果实验显示 RK-FAC 在 Long 上没有显著改善，说明 Q 在 VLA setting 下确实不够准确。这时可以 fallback 到 FPI-style weighted regression + frozen-VLM V 的组合。

---

## 八、论文结构

**Title**: *Residual-Kinetic Flow Actor-Critic: OOD-Robust Reinforcement Learning for Vision-Language-Action Models*

### Structure (9 pages + appendix)

**1. Introduction** (1.5 pages)
- 开头：VLA RL 的不可能三角（temporal credit vs OOD vs efficiency）
- 核心 insight：残差动能 = likelihood-free KL bound + OOD auto-fallback
- Contributions 列表

**2. Preliminaries** (1 page)
- Flow matching + VLA architecture
- FLAC's GSB framework
- OOD problem in VLA RL（pi-StepNFT vs PPO 数据）

**3. Residual-Kinetic Flow Actor-Critic** (2.5 pages)
- 3.1 Residual kinetic energy definition + Theorem 1
- 3.2 Frozen-VLM Q design + Theorem 3
- 3.3 Actor-Critic update + Automatic α
- 3.4 Algorithm 1
- 3.5 Implementation details

**4. Theoretical Analysis** (1.5 pages)
- Theorem 4: Improvement guarantee
- Theorem 5: OOD robustness guarantee
- Theorem 6: Unification with FLAC/SAC/pi-StepNFT

**5. Experiments** (2.5 pages)
- 5.1 Main results: LIBERO (Short + Long) + ManiSkill (IND + OOD)
- 5.2 OOD analysis: frozen vs trainable critic
- 5.3 Ablations: trust region types, α tuning, NFE, replay buffer
- 5.4 Training efficiency: wall-clock time comparison

**6. Related Work** (0.5 pages)

**7. Conclusion** (0.5 pages)

**Appendix**:
- A: Full proofs of Theorems 1-6
- B: Extended experimental results
- C: Hyperparameter sensitivity
- D: Detailed comparison with FLAC

---

## 九、时间线

| 时间 | 里程碑 |
|------|--------|
| 2026.04 W1-2 | Theorem 1-6 完整证明；实现 residual KE 计算 + frozen-VLM Q |
| 2026.04 W3-4 | LIBERO-Short 验证 Q 质量 + actor 稳定性 |
| 2026.05 W1-2 | LIBERO-Long 验证 temporal credit |
| 2026.05 W3-4 | ManiSkill OOD 全面验证 |
| 2026.06 W1-2 | 消融实验 |
| 2026.06 W3-4 | 论文写作 |
| 2026.07 W1 | 内部 review + 修改 |
| 2026.07 底 | 提交 NeurIPS 2026 |

**硬件需求**：32×A100 足够。LIBERO 用 8×A100（rollout + 训练），ManiSkill 用 24×A100（4096 并行环境）。

---

## 十、风险评估

| 风险 | 等级 | Mitigation |
|------|------|-----------|
| Q-gradient through ODE 不稳定 | **中** | SAC Flow 已验证；残差 KE 天然限制梯度；gradient clipping |
| Frozen-VLM Q 不够 expressive | **中** | 消融 MLP 层数；fallback 到 FPI-style weighted regression |
| OOD 改善不显著 | **中** | Theorem 5 提供理论预期；消融 frozen vs trainable 定位原因 |
| Reviewer 说"就是 FLAC for VLA" | **高** | 强调残差 KE（非零 drift 参考）的理论新意 + OOD guarantee + frozen-VLM Q 设计 |
| pi-StepNFT 消融暗示 learned signal 无效 | **中** | RK-FAC 的信息利用方式不同（Q-gradient vs y sign）；如失败则 pivot 到 FPI |
| 某组抢发类似 idea | **低** | FLAC→VLA 的路径需要非平凡的理论扩展（残差 KE + OOD bound），壁垒较高 |

---

## 十一、一句话总结

> **Flow VLA fine-tuning 的信任域应该在路径空间的残差动能中定义——它精确等于与预训练策略的 KL 散度，在 OOD 状态下自动回退到预训练行为，且不需要计算任何似然。结合 frozen-VLM Q-network 提供 temporal credit，RK-FAC 是第一个同时在 Long-horizon 和 OOD 上达到 competitive 的 flow VLA RL 方法。**
