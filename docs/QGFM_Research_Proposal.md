# Q-Guided Flow Matching (QGFM)：通过 Q-Gradient Action Perturbation 实现 Flow VLA 的简洁高效 RL

## NeurIPS 2026 完整研究方案

---

## 一、Motivation：三种 Gradient Carrier 的困境

### 1.1 Flow VLA RL 的核心技术挑战

截至 2026 年 3 月，Flow VLA + RL 领域的所有方法可以按 **gradient carrier**（梯度信号从何而来）精确分类为三大阵营。每个阵营都有自己的结构性缺陷：

**阵营 A：CFM 拟合误差 $(v_\theta - u)$**

代表：FPI、AWM、RWFM、ORW-CFM、FPO（作为 proxy ratio）

$$\nabla_\theta \mathcal{L} = w \cdot 2(v_\theta - u) \cdot \nabla_\theta v_\theta$$

问题：on-policy pretrained VLA 上 $v_\theta \approx u$ → gradient carrier $(v_\theta - u) \approx 0$ → **收敛极慢**。AWM 实验验证：曲线从 0.35 涨到 0.6 后接近停滞。FPI 实验验证：收敛慢。

**阵营 B：SDE noise residual $e_t$**

代表：πRL Flow-SDE（PPO）、πRL Flow-Noise、pi-StepNFT、Flow-GRPO

$$\nabla_\theta \mathcal{L} \propto f(\hat{A}_t) \cdot \Sigma_t^{-1} e_t \cdot B_t \cdot \nabla_\theta v_\theta$$

优势：$e_t$ 由注入的 SDE noise 产生，始终非零 → gradient 不 vanish → **收敛快**。

代价：
- **必须做 SDE rollout**（ODE→SDE 转换引入 discretization error，πRL Tab.6 显示 train performance 下降）
- **必须算 importance ratio**（SDE transition density ratio，Eq.15）
- **MDP horizon 膨胀**（two-layer MDP 的 effective horizon = env steps × denoising steps）
- 系统复杂度高（noise schedule tuning、mixed ODE-SDE sampling、ratio clipping）

**阵营 C：Q-gradient through ODE**

代表：SAC Flow、RK-FAC、QAM

$$\nabla_\theta \mathcal{L} = \nabla_a Q \cdot \frac{\partial a}{\partial \theta} = \nabla_a Q \cdot \prod_{k=1}^{K} \frac{\partial x_{k+1}}{\partial x_k} \cdot \nabla_\theta v_\theta$$

优势：不需要 SDE，不需要 ratio。

代价：gradient 穿过 K 步 ODE 的 chain rule → **收敛慢**（RK-FAC 实验验证：能收敛但慢）。K 步 Jacobian 乘积导致梯度方向不稳定。

### 1.2 核心 Insight

**三个阵营的根本矛盾**：

| | Gradient 强度 | 不需要 SDE | 不需要 BPTT | 系统简洁性 |
|---|:-:|:-:|:-:|:-:|
| 阵营 A (CFM) | ✗ vanish | ✓ | ✓ | ✓ |
| 阵营 B (SDE) | ✓ | ✗ | ✓ | ✗ |
| 阵营 C (BPTT) | ✓ | ✓ | ✗ | ✗ |

**QGFM 的核心 idea**：不把 Q-gradient 反传穿过 ODE（阵营 C），而是用 Q-gradient **扰动 flow matching 的 target action**，然后做标准 BC。

$$a' = a + \eta \cdot \nabla_a Q(s, a)$$

$$\mathcal{L}_{\text{QGFM}} = \|v_\theta(x_\tau, \tau | s) - (a' - \varepsilon)\|^2$$

此时 gradient carrier 变为：

$$v_\theta - u' = (v_\theta - u) - \eta \nabla_a Q \approx -\eta \nabla_a Q$$

**Q-gradient 不穿过 ODE（不做 BPTT），而是通过 perturbed target 隐式传递给 flow matching loss。**

| | Gradient 强度 | 不需要 SDE | 不需要 BPTT | 系统简洁性 |
|---|:-:|:-:|:-:|:-:|
| **QGFM** | **✓** ($\eta \nabla_a Q$) | **✓** | **✓** | **✓** |

**一句话 message**：*不要让 RL 的梯度穿过 flow matching 的 ODE——让 RL 告诉 flow matching "target 应该往哪移"，然后让 flow matching 自己学。*

---

## 二、相关工作全景与精确定位

### 2.1 按技术路线分类

#### 阵营 A：Weighted / Reward-Conditioned Flow Matching

| 论文 | 时间 | 核心方法 | 收敛性 |
|------|------|---------|-------|
| FPI (本组) | 2026 | $w \cdot \|v_\theta - u\|^2$, iterative | 慢（$(v_\theta-u)\to 0$） |
| AWM (Xue et al.) | 2025.09 | Advantage-weighted FM | 慢（实验验证） |
| RWFM (Pfrommer et al.) | 2025.07 | Single-round reward-weighted FM | 单轮，无迭代 |
| ORW-CFM (ICLR'25) | 2024.10 | Online reward-weighted + W₂正则 | 连续时间，W₂防collapse |

共同问题：gradient carrier $(v_\theta - u)$ on-policy vanish。

#### 阵营 B：SDE Density Ratio + Policy Gradient

| 论文 | 时间 | 核心方法 | Venue |
|------|------|---------|-------|
| πRL Flow-SDE | 2025.10 | SDE transition density + PPO | Preprint |
| πRL Flow-Noise | 2025.10 | Learnable noise net + joint log-prob + PPO | Preprint |
| pi-StepNFT | 2026.03 | SDE density + contrastive mirror | Preprint |
| Flow-GRPO | 2025.05 | ODE→SDE + GRPO (image gen) | Preprint |
| ReinFlow | 2025.05 | Learnable noise + discrete MDP | NeurIPS 2025 |

共同特征：需要 SDE rollout + noise schedule tuning。系统复杂。

#### 阵营 C：Q-Gradient Through ODE

| 论文 | 时间 | 核心方法 | 收敛性 |
|------|------|---------|-------|
| SAC Flow | 2026.01 | GRU/Transformer 重参数化 + SAC | 收敛但需特殊架构 |
| RK-FAC (本组) | 2026 | Residual kinetic energy + Q-grad | 慢 |
| QAM | 2026.01 | Adjoint Matching + TD Q | Off-policy，需 adjoint ODE |
| FLAC | 2026.02 | Kinetic energy + Q-grad | From-scratch only |

共同特征：需要 BPTT through ODE solver。

#### 阵营 D：黑盒 / 不改权重

| 论文 | 时间 | 核心方法 |
|------|------|---------|
| DSRL | 2025.06 | Noise 空间 SAC，冻结 VLA |
| PA-RL | 2024.12 | Q-optimized action → SFT 蒸馏 |

### 2.2 QGFM 的精确定位

QGFM 不属于以上任何阵营。它是一种新的范式：

> **Q-gradient 不穿过 ODE（区别于阵营 C），不用 SDE noise 做 gradient carrier（区别于阵营 B），不依赖 $(v_\theta - u)$（区别于阵营 A），而是用 Q-gradient 扰动 target action 后做标准 flow matching。**

### 2.3 与 PA-RL 的精确区别

PA-RL（Mark et al. 2024）是最接近的已有工作。核心区别：

| 维度 | PA-RL | QGFM |
|------|-------|------|
| Action 优化 | Multi-step gradient ascent to $\arg\max_a Q$ | **One-step perturbation** $a + \eta \nabla_a Q$ |
| 偏离程度 | 大（到 Q 的 argmax） | 小（一步 gradient step） |
| 蒸馏方式 | 标准 SFT loss | **Flow matching loss（利用 $v_\theta \approx u$ 的特殊性质）** |
| Iterative? | 否（Q 固定，一轮蒸馏） | **是（Q 和 policy 交替更新）** |
| 为 flow VLA 设计？ | 否（通用 policy class） | **是（利用 flow matching 的数学结构）** |
| 理论分析 | 无 | **$\nabla_\theta \mathcal{L} \approx -2\eta \nabla_a Q \cdot \nabla_\theta v_\theta$ 的精确推导** |
| 收敛速度 | 快但不稳定（大偏移） | 稳定且快（小偏移 + iterative） |

PA-RL 的问题：multi-step 到 argmax 偏离原始 policy 太远 → 蒸馏后的 policy 可能在原始 state distribution 上行为差。ConRFT 中 PA-RL 只达到 71.3%。

QGFM 的 one-step perturbation 保证 $a'$ 在 $a$ 的邻域内 → flow matching 可以稳定拟合 → iterative improvement 不崩。

### 2.4 与 QAM 的区别

QAM（Adjoint Matching + TD Q，2026.01）也用 Q，但：

| 维度 | QAM | QGFM |
|------|-----|------|
| Q 的使用方式 | Q-gradient 通过 adjoint ODE 反传到 $\theta$ | Q-gradient 只扰动 target $a'$，不穿过 ODE |
| 需要 adjoint？ | ✓（额外 backward ODE solve） | **✗** |
| 训练稳定性 | 依赖 adjoint 精度 | **标准 BC，天然稳定** |
| Off-policy？ | ✓ | **✓**（replay buffer） |

---

## 三、方法：Q-Guided Flow Matching (QGFM)

### 3.1 Problem Setup

**环境**：MDP $(S, A, P, R, \gamma)$，$A \subseteq \mathbb{R}^{d_a}$ 连续动作空间。

**初始策略**：$\pi_{\text{pre}}$ = pre-trained flow matching VLA（如 π₀/π₀.₅），velocity field $v_{\theta_0}(x_\tau, \tau \mid o, \ell)$。

**目标**：通过 iterative improvement 最大化任务成功率。

**架构**（和 πRL 完全一致）：
- Frozen VLM backbone（PaliGemma-3B）
- Trainable flow action expert（~300M 参数）
- Q network：$Q_\phi(h_{\text{VLM}}, a)$，~5-10M 参数，两个 Q head
- 只更新 action expert 和 Q network

### 3.2 核心数学推导

#### 3.2.1 标准 Flow Matching Loss 的 On-Policy Gradient

标准 CFM loss 在 rollout data $(s, a)$ 上：

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{\tau, \varepsilon}\left[\|v_\theta(x_\tau, \tau | s) - u\|^2\right], \quad u = a - \varepsilon, \quad x_\tau = \tau a + (1-\tau)\varepsilon$$

梯度：

$$\nabla_\theta \mathcal{L}_{\text{CFM}} = 2\mathbb{E}_{\tau,\varepsilon}\left[(v_\theta - u) \cdot \nabla_\theta v_\theta\right]$$

当 $a \sim \pi_\theta$（on-policy data）时，$v_\theta$ 已经被训练去预测 $u = a - \varepsilon$。因此 $v_\theta(x_\tau, \tau | s) \approx u$，gradient carrier $(v_\theta - u) \approx 0$。

#### 3.2.2 Q-Guided Target Perturbation

定义 perturbed action：

$$a' = a + \eta \cdot \frac{\nabla_a Q_\phi(s, a)}{\|\nabla_a Q_\phi(s, a)\| + \epsilon}$$

其中 $\eta > 0$ 是 perturbation step size，$\epsilon$ 是数值稳定项。归一化保证 perturbation 幅度恒定为 $\eta$（不受 Q-gradient 的 scale 影响）。

对应的 perturbed target：

$$u' = a' - \varepsilon = u + \eta \cdot \hat{g}, \quad \hat{g} = \frac{\nabla_a Q_\phi(s, a)}{\|\nabla_a Q_\phi(s, a)\| + \epsilon}$$

QGFM loss：

$$\mathcal{L}_{\text{QGFM}} = \mathbb{E}_{\tau, \varepsilon}\left[\|v_\theta(x_\tau, \tau | s) - u'\|^2\right]$$

#### 3.2.3 Gradient 分析（Theorem 1）

**Theorem 1 (Non-Vanishing Gradient)**. 在 on-policy data 上（$v_\theta \approx u$），QGFM loss 的梯度为：

$$\nabla_\theta \mathcal{L}_{\text{QGFM}} = 2\mathbb{E}_{\tau,\varepsilon}\left[(v_\theta - u') \cdot \nabla_\theta v_\theta\right]$$

$$\approx 2\mathbb{E}_{\tau,\varepsilon}\left[(v_\theta - u - \eta\hat{g}) \cdot \nabla_\theta v_\theta\right]$$

$$\approx 2\mathbb{E}_{\tau,\varepsilon}\left[\underbrace{(v_\theta - u)}_{\approx 0} \cdot \nabla_\theta v_\theta - \eta\hat{g} \cdot \nabla_\theta v_\theta\right]$$

$$\approx -2\eta \mathbb{E}_{\tau,\varepsilon}\left[\hat{g} \cdot \nabla_\theta v_\theta\right]$$

**即：QGFM 的梯度方向由 $\nabla_a Q$ 决定，大小由 $\eta$ 控制，始终非零。**

更准确地：

$$\nabla_\theta \mathcal{L}_{\text{QGFM}} = \underbrace{2\mathbb{E}[(v_\theta - u) \cdot \nabla_\theta v_\theta]}_{\text{标准 BC gradient（小）}} - \underbrace{2\eta\mathbb{E}[\hat{g} \cdot \nabla_\theta v_\theta]}_{\text{Q-guided improvement（不 vanish）}}$$

#### 3.2.4 与 Policy Gradient 的联系（Theorem 2）

**Theorem 2 (Implicit Policy Gradient)**. QGFM 的 gradient 与以下 policy gradient 的 projected form 等价：

$$-2\eta\mathbb{E}\left[\hat{g} \cdot \nabla_\theta v_\theta\right] = -2\eta\mathbb{E}\left[\nabla_\theta \langle v_\theta, \hat{g} \rangle\right]$$

而 $\langle v_\theta, \hat{g} \rangle$ 是 velocity field 在 Q-gradient 方向的投影。最小化 QGFM loss = 让 velocity field 在每个 $(x_\tau, \tau)$ 处向 Q 增大的方向偏移。

**含义**：QGFM 是一种 **implicit policy gradient** 方法——它不显式计算 $\log \pi$，但通过 target perturbation 隐式实现了沿 Q-gradient 方向的 policy improvement。

#### 3.2.5 Interpolation 点的处理

一个微妙的问题：$x_\tau = \tau a + (1-\tau)\varepsilon$ 是基于原始 $a$ 的插值，还是基于 $a'$ 的插值？

**选择 1**：$x_\tau = \tau a + (1-\tau)\varepsilon$（原始插值，只改 target）

优势：rollout 时不需要 $a'$，$a'$ 只在 loss 计算时需要。更简单。

$$\mathcal{L} = \|v_\theta(\tau a + (1-\tau)\varepsilon, \tau | s) - (a' - \varepsilon)\|^2$$

**选择 2**：$x_\tau' = \tau a' + (1-\tau)\varepsilon$（perturbed 插值，改 target 和 input）

优势：flow matching 的 conditional path 一致性更好。

$$\mathcal{L} = \|v_\theta(\tau a' + (1-\tau)\varepsilon, \tau | s) - (a' - \varepsilon)\|^2$$

但 $x_\tau'$ 与 $x_\tau$ 的差异 $= \tau \eta \hat{g}$，当 $\eta$ 小时差异小。

**推荐选择 1**（更简单，$\eta$ 小时两者近似相等）。消融实验对比两者。

### 3.3 Q-Network 设计

#### 3.3.1 Architecture

$$Q_\phi(s, a) = \text{MLP}_\phi(\text{sg}[h_{\text{VLM}}(o, \ell)], \; a)$$

- 输入：frozen VLM features（stop gradient）+ action
- 两个独立的 Q network $Q_{\phi_1}, Q_{\phi_2}$（Double Q，减少过估计）
- 每个 Q：3 层 MLP，hidden dim 256，~1M 参数
- Target networks $\bar{Q}_1, \bar{Q}_2$ 用 EMA 更新

#### 3.3.2 TD Learning

$$y = r + \gamma \min_{j=1,2} \bar{Q}_j(s', a'), \quad a' \sim \pi_\theta(\cdot|s')$$

$$\mathcal{L}_Q = \frac{1}{2}\sum_{j=1}^{2}\mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\left[(Q_{\phi_j}(s,a) - y)^2\right]$$

$a'$ 由当前 policy 的 ODE rollout 产生（确定性，不需要 SDE）。

#### 3.3.3 Replay Buffer

维护 replay buffer $\mathcal{B}$，存储 $(o_t, \ell, a_t, r_t, o_{t+1})$。Off-policy 数据复用。Q-learning 天然 off-policy。

### 3.4 完整算法

---

#### Algorithm 1: Q-Guided Flow Matching (QGFM)

**Input:** Pre-trained flow VLA $\pi_{\theta_0}$，Q networks $Q_{\phi_1}, Q_{\phi_2}$，target networks $\bar{Q}_1, \bar{Q}_2$，replay buffer $\mathcal{B}$，perturbation step size $\eta$

**For** iteration $k = 0, 1, \ldots, K-1$ **do:**

**Step 1 (ODE Rollout):** 用 $\pi_{\theta_k}$ 做 **确定性 ODE rollout**（不需要 SDE），收集 transitions $(o_t, \ell, a_t, r_t, o_{t+1})$ 存入 $\mathcal{B}$

**Step 2 (Q Update):** 从 $\mathcal{B}$ 采样 mini-batch，更新 Q（可做多步）：

$$y = r + \gamma \min_j \bar{Q}_j(s', \pi_{\theta_k}(s'))$$

$$\phi_j \leftarrow \phi_j - \alpha_Q \nabla_{\phi_j} \frac{1}{2}(Q_{\phi_j}(s,a) - y)^2, \quad j = 1, 2$$

**Step 3 (Target Perturbation):** 对 rollout data 中每个 $(s_t, a_t)$：

$$\hat{g}_t = \frac{\nabla_a \min_j Q_{\phi_j}(s_t, a_t)}{\|\nabla_a \min_j Q_{\phi_j}(s_t, a_t)\| + \epsilon}$$

$$a_t' = a_t + \eta \cdot \hat{g}_t$$

**Step 4 (Q-Guided Flow Matching Update):** 在 perturbed data $(s_t, a_t')$ 上做标准 flow matching：

$$\mathcal{L}_{\text{QGFM}}(\theta) = \mathbb{E}_{(s,a') \sim \mathcal{D}_k, \; \tau \sim U[0,1], \; \varepsilon \sim \mathcal{N}(0,I)}\left[\|v_\theta(x_\tau, \tau | s) - (a' - \varepsilon)\|^2\right]$$

$$x_\tau = \tau a + (1-\tau)\varepsilon$$

多 epoch 梯度下降：$\theta_{k+1} \leftarrow \theta_k - \alpha_\pi \nabla_\theta \mathcal{L}_{\text{QGFM}}$

**Step 5 (Target Network Update):**

$$\bar{\phi}_j \leftarrow \rho \bar{\phi}_j + (1-\rho)\phi_j, \quad j = 1, 2$$

**End For**

**Return** $\pi_{\theta_K}$

---

### 3.5 关键实现细节

#### 3.5.1 兼容 π₀/π₀.₅

和 πRL 完全一致的架构：冻结 VLM，只训 action expert + Q MLP。QGFM 不修改 VLA 的任何组件——只是在训练时把 flow matching 的 target action 从 $a$ 换成 $a'$。

#### 3.5.2 $\eta$ 的选择与调度

$\eta$ 控制 perturbation 幅度。太大 → $a'$ 偏离 $\pi_\theta$ 的支撑集 → flow matching 拟合不稳定。太小 → 接近 BC → 改善慢。

**推荐**：$\eta$ 随训练递增。
- 早期 Q 不准 → $\eta$ 小 → 接近 BC → 安全启动
- 后期 Q 变准 → $\eta$ 大 → 更 aggressive improvement

具体 schedule：$\eta_k = \eta_0 + (\eta_{\max} - \eta_0) \cdot \min(1, k / K_{\text{warmup}})$

$\eta_0 = 0.001$, $\eta_{\max} = 0.01 \sim 0.05$（action space 的 scale 决定），$K_{\text{warmup}} = 50$。

#### 3.5.3 Q Warm-Up

前 $N_{\text{warmup}}$ 个 iteration 设 $\eta = 0$（纯 BC），只训 Q。等 Q 的 TD error 收敛后再启用 perturbation。这保证 perturbation 方向是有意义的。

与 FPI 的 "value warm-up" 理念相同。

#### 3.5.4 Gradient Clipping on Perturbation

除了归一化 $\nabla_a Q$，还可以 clip perturbation 到 action space 的合理范围：

$$a' = \text{clip}(a + \eta \hat{g}, \; a_{\min}, \; a_{\max})$$

防止 $a'$ 超出 action space boundaries。

#### 3.5.5 多 Epoch Flow Matching

因为 QGFM 的 loss 是标准 regression（不涉及 importance ratio），可以在同一批 perturbed data 上做多 epoch 训练。这比 on-policy PPO（每批数据只能用 1-2 次）有更高的 data utilization。

但需要注意：多 epoch 后 $v_\theta \to u'$，gradient carrier 又变小了。所以建议 2-4 epochs per iteration。

#### 3.5.6 和 πRL codebase 的兼容性

QGFM 可以直接在 πRL (RLinf) codebase 上实现：

- Rollout 部分：去掉 SDE noise injection，用纯 ODE rollout（更简单、更快）
- Q network：和 πRL 的 critic 架构相同，但输入加 action（Q(s,a) 而非 V(s)）
- Policy update：把 PPO loss 替换为 perturbed flow matching loss
- 移除：importance ratio 计算、PPO clip、GAE 计算
- 新增：$\nabla_a Q$ 计算 + target perturbation

---

## 四、理论分析

### 4.1 Theorem 1: Non-Vanishing Gradient（已在 3.2.3 给出）

QGFM 的 gradient $\approx -2\eta \hat{g} \cdot \nabla_\theta v_\theta$，只要 $\eta > 0$ 且 $\nabla_a Q \neq 0$，gradient 非零。

### 4.2 Theorem 2: Implicit Policy Gradient（已在 3.2.4 给出）

QGFM 隐式执行沿 Q-gradient 方向的 policy improvement。

### 4.3 Theorem 3: Monotonic Improvement Guarantee

**定理3.** 设 $Q_\phi$ 的估计误差 $\|Q_\phi - Q^{\pi_k}\|_\infty \leq \epsilon_Q$，flow matching 拟合误差 $\mathbb{E}_s[W_2(\hat{\pi}_{k+1}, \pi_{k+1}^*)] \leq \epsilon_{\text{FM}}$。则经过一步 QGFM 更新：

$$V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s) + \frac{\eta}{1-\gamma}\mathbb{E}_{a \sim \pi_k}\left[\|\nabla_a Q^{\pi_k}(s,a)\|\right] - \frac{C_1}{(1-\gamma)^2}\epsilon_Q - \frac{C_2}{(1-\gamma)^2}\epsilon_{\text{FM}} - O(\eta^2)$$

**含义**：
- 第一项是 improvement：$\nabla_a Q$ 的 norm 越大（当前 action 越不是 local optimum），improvement 越大
- 第二、三项是误差：Q 估计误差和 flow matching 拟合误差
- $O(\eta^2)$：perturbation 的二阶误差，$\eta$ 小时可忽略

**证明思路**：
1. $a' = a + \eta \nabla_a Q$ 的 Q 值：$Q(s, a') \approx Q(s, a) + \eta \|\nabla_a Q\|^2 + O(\eta^2)$（一阶 Taylor）
2. Flow matching 在 $a'$ 上训练 → $\pi_{k+1}$ 在 $a'$ 附近有高概率 → $\mathbb{E}_{a \sim \pi_{k+1}}[Q(s,a)] \approx Q(s, a')$
3. Performance difference lemma 给出 $V^{\pi_{k+1}} - V^{\pi_k}$ 的 bound

### 4.4 Theorem 4: Convergence Rate

**定理4.** 在 bounded reward, Lipschitz Q, bounded $\|\nabla_a Q\|$ 假设下，经过 $K$ 轮 QGFM：

$$V^* - V^{\pi_K} \leq \frac{\gamma^K}{1-\gamma} R_{\max} + \frac{C}{(1-\gamma)^2}(\epsilon_Q + \epsilon_{\text{FM}}) + O(K\eta^2)$$

第一项以 $\gamma^K$ 指数衰减。第三项随 $K$ 线性增长——这约束了 $\eta$ 不能太大（否则二阶误差累积）。实践中用小 $\eta$ + 多轮迭代。

### 4.5 Theorem 5: 与 PA-RL, AWM, πRL 的统一视角

**(a)** PA-RL = QGFM 取 $\eta \to \infty$（multi-step 到 argmax），非 iterative（$K=1$）。

**(b)** AWM = QGFM 取 $\eta = 0$，用 advantage weight 替代 perturbation。

**(c)** πRL Flow-SDE PPO ≈ QGFM 在 SDE regime 下的 first-order 等价：两者的 effective gradient 都是 $\nabla_a Q \cdot \nabla_\theta v_\theta$（πRL 通过 SDE ratio + advantage，QGFM 通过 target perturbation），但实现路径完全不同。

---

## 五、QGFM vs 所有现有方法的系统对比

### 5.1 技术难题的解决方式

| 技术挑战 | 阵营 B (πRL) | 阵营 C (SAC Flow) | **QGFM** |
|---------|------------|----------------|---------|
| Log-likelihood | SDE transition density | 不需要（Q-grad） | **不需要** |
| SDE 必要性 | ✓ 必须做 SDE rollout | ✗ | **✗ 纯 ODE** |
| BPTT 必要性 | ✗ | ✓ 穿过 ODE | **✗** |
| Importance ratio | ✓ 必须算 | ✗ | **✗** |
| MDP horizon | 膨胀（two-layer） | 不膨胀 | **不膨胀** |
| Off-policy | ✗ (on-policy) | ✓ | **✓ (replay buffer)** |
| 训练 loss | PPO surrogate | SAC objective | **标准 flow matching MSE** |

### 5.2 全方法对比表

| 方法 | Gradient carrier | SDE? | Ratio? | BPTT? | Off-policy? | Simplicity |
|------|-----------------|:----:|:------:|:-----:|:-----------:|:----------:|
| πRL PPO | $e_t$ | ✓ | ✓ | ✗ | ✗ | 中 |
| pi-StepNFT | $e_t$ | ✓ | ✗ | ✗ | ✗ | 中 |
| FPO++ | $(v_\theta-u)$ | ✗ | ✓ | ✗ | ✗ | 中 |
| SAC Flow | $\nabla_a Q$ thru ODE | ✗ | ✗ | ✓ | ✓ | 低 |
| FPI/AWM | $(v_\theta-u)$ | ✗ | ✗ | ✗ | Partial | 高 |
| PA-RL | $a^*-a$ | ✗ | ✗ | ✗ | ✗ | 高 |
| **QGFM** | **$\eta \nabla_a Q$** | **✗** | **✗** | **✗** | **✓** | **高** |

### 5.3 训练流程简洁性对比

**πRL Flow-SDE PPO**:
```
1. SDE Rollout (需调 noise level, ODE-SDE conversion)
2. 计算 SDE transition density ratio (Eq.9 + Eq.15)
3. 计算 GAE (需训 critic V)
4. PPO clip surrogate
5. 多 epoch update (需 clip 保证稳定)
```

**QGFM**:
```
1. ODE Rollout (确定性，无 noise schedule)
2. TD update Q (标准 off-policy)
3. 计算 ∇_a Q, perturb target: a' = a + η·ĝ
4. 标准 flow matching loss on (s, a')
5. 多 epoch update (BC 天然稳定)
```

QGFM 去掉了：SDE、importance ratio、PPO clip、GAE。核心操作只有 Q-learning + perturbed BC。

---

## 六、实验方案

### 6.1 Research Questions

| RQ | 问题 | 验证方式 |
|----|------|---------|
| RQ1 | QGFM 是否 match πRL PPO 的最终性能？ | LIBERO 四个 suite 的 success rate |
| RQ2 | QGFM 的收敛速度和 πRL PPO 相比如何？ | Training curves (success rate vs step) |
| RQ3 | QGFM 是否比 AWM/FPI 收敛更快？ | Head-to-head training curves |
| RQ4 | Off-policy data reuse 是否提升 sample efficiency？ | Wall-clock time 对比 |
| RQ5 | Perturbation 的 gradient carrier 是否确实比 $(v_\theta - u)$ 更大？ | Gradient norm 统计 |
| RQ6 | QGFM 在 ManiSkill 大规模多任务上是否有效？ | ManiSkill 4352 task success rate |

### 6.2 Benchmarks

| Benchmark | Tasks | 选择理由 |
|-----------|-------|---------|
| LIBERO-Spatial | 10 tasks | 短时域，基本验证 |
| LIBERO-Object | 10 tasks | 物体泛化 |
| LIBERO-Goal | 10 tasks | 目标泛化 |
| LIBERO-Long | 10 tasks | **Long-horizon（temporal credit 验证）** |
| ManiSkill Generalization | 4352 tasks | 大规模多任务 |
| ManiSkill OOD | 3 types | OOD 泛化 |

### 6.3 Baselines

| 方法 | 为什么必须对比 |
|------|---------------|
| **πRL Flow-SDE PPO** | **直接竞争者**：当前 SOTA |
| **πRL Flow-Noise PPO** | **直接竞争者**：πRL 的另一方案 |
| **πRL GRPO** | 无 critic baseline |
| **pi-StepNFT** | 无 critic + contrastive mirror |
| **SFT baseline** | 不做 RL |
| **AWM/FPI** | 阵营 A baseline（验证 QGFM 比 weighted FM 快） |

### 6.4 Ablation Studies

| 消融 | 对比 | 验证什么 |
|------|------|---------|
| **$\eta$** | 0.001, 0.005, 0.01, 0.05, 0.1 | Perturbation 幅度的影响 |
| **$\eta$ 调度 vs 固定** | 递增 vs 固定 | 调度策略 |
| **Q warm-up 步数** | 0, 10, 50, 100 | Q 质量对启动的影响 |
| **Interpolation 点** | 选择 1 vs 选择 2 | $x_\tau$ 的构造方式 |
| **Update epochs** | 1, 2, 4 | 多 epoch 复用 |
| **Q architecture** | 2层 vs 3层 vs 4层 MLP | Q 容量 |
| **Normalized vs raw $\nabla_a Q$** | 归一化 vs 不归一化 | Perturbation 方向 vs 方向+大小 |
| **QGFM vs PA-RL** | One-step vs multi-step perturbation | Conservative vs aggressive |
| **Gradient norm 统计** | 比较 $\|v_\theta - u\|$ vs $\|\eta \hat{g}\|$ vs $\|e_t\|$ | 三种 carrier 的强度 |

### 6.5 预期结果

| | Spatial | Object | Goal | Long | Avg |
|---|:-:|:-:|:-:|:-:|:-:|
| SFT (π₀) | 65.3 | 64.4 | 49.8 | 51.2 | 57.6 |
| πRL PPO (π₀) | 98.4 | 99.4 | 96.2 | 90.2 | 96.1 |
| πRL GRPO (π₀) | 97.8 | 97.8 | 83.2 | 81.4 | 90.0 |
| **QGFM (π₀)** | **~97** | **~99** | **~93** | **~88-92** | **~94-96** |

QGFM 预期在 GRPO（无 temporal credit）和 PPO（full temporal credit）之间，偏向 PPO 端（因为 Q 提供 per-step gradient direction）。

---

## 七、应对 Reviewer 质疑

### Q1: "这不就是 PA-RL 吗？"

**回应**：

1. PA-RL 做 **multi-step gradient ascent to argmax**（偏离大），QGFM 做 **one-step perturbation**（保守）。ConRFT 实验显示 PA-RL 只达 71.3%（偏离太远的后果）。
2. PA-RL 是 **非 iterative** 的（Q 固定，一轮蒸馏），QGFM 是 **iterative** 的（Q 和 policy 交替更新）。
3. QGFM 有精确的数学推导（Theorem 1-4）表明 on-policy pretrained VLA 上 $(v_\theta - u) \approx 0$ 时 perturbation 提供了唯一的有效 gradient carrier。PA-RL 没有这个分析。
4. 消融实验直接对比 one-step vs multi-step perturbation。

### Q2: "Q 不准的话 perturbation 方向错了怎么办？"

**回应**：

1. $\eta$ 控制 perturbation 幅度。Q 不准时 $\eta$ 小 → $a' \approx a$ → 退化为 BC → 不会比 SFT 差。
2. $\eta$ warm-up 保证 Q 收敛后再加大 perturbation。
3. 归一化 $\nabla_a Q$ 保证 perturbation 方向有意义（即使 magnitude 不准）。
4. Double Q + target network 是标准的 Q 稳定化技术。
5. Theorem 3 的 bound 显式包含 $\epsilon_Q$——Q 误差有界时 improvement 有界。

### Q3: "为什么不直接用 πRL PPO？QGFM 优势在哪？"

**回应**：

1. **简洁性**：QGFM 不需要 SDE rollout、importance ratio、PPO clip、GAE。核心只有 Q-learning + perturbed BC。
2. **Off-policy**：πRL PPO 是 on-policy（每批数据用 1-2 次），QGFM 用 replay buffer + 多 epoch。
3. **不需要 noise schedule tuning**：πRL 的 Tab.6 显示 noise level 敏感（0.2 不稳定，0.8 train 差），QGFM 用纯 ODE 不涉及。
4. **不膨胀 MDP horizon**：πRL 的 two-layer MDP 把 effective horizon 乘以 denoising steps。
5. **理论分析**：QGFM 有 monotonic improvement guarantee（Theorem 3），πRL PPO 没有（PPO 的 theory 依赖 exact log-prob，SDE proxy 的 theory gap 未被 address）。

### Q4: "和 QAM 的区别？"

**回应**：

QAM 用 Q-gradient 通过 **adjoint ODE** 反传到 $\theta$。QGFM 用 Q-gradient **扰动 target**，不穿过任何 ODE。

| | QAM | QGFM |
|---|---|---|
| Q 如何影响 policy | $\nabla_\theta Q$ via adjoint ODE | $\nabla_a Q$ perturbs target |
| 需要 adjoint solver？ | ✓ | **✗** |
| 需要 BPTT？ | ✓（adjoint = backward ODE） | **✗** |
| 训练 loss | Adjoint matching | **标准 flow matching MSE** |

### Q5: "Novelty 够不够？不就是给 target 加了个 perturbation？"

**回应**：

1. **诊断贡献**（Section 3）：首次形式化三类 gradient carrier 的分类（CFM vanish / SDE noise / Q-BPTT），解释了为什么 FPI 慢、为什么 πRL 需要 SDE、为什么 SAC Flow 需要 BPTT。
2. **理论贡献**（Section 4）：Theorem 1 的 non-vanishing proof、Theorem 2 的 implicit policy gradient equivalence、Theorem 5 的统一视角。
3. **实践贡献**：最简洁的 flow VLA RL 方法——不需要 SDE、不需要 ratio、不需要 BPTT、不需要 PPO clip，性能 comparable。
4. "给 target 加 perturbation" 看似简单，但**它为什么 work 的数学解释**（on-policy vanishing + Q-perturbation restores gradient）是 non-trivial 的 insight。

---

## 八、论文结构

**Title**: *Q-Guided Flow Matching: Simple and Effective Reinforcement Learning for Flow-Based VLA*

**Subtitle (optional)**: *Policy Improvement via Target Perturbation Without SDE, Likelihood, or Backpropagation Through ODE*

### Structure (9 pages + appendix)

**1. Introduction** (1.5 pages)
- 开头：flow VLA RL 的三类 gradient carrier 及各自的困境
- 核心 insight：Q-gradient perturb target → 第四种 non-vanishing carrier
- Key message + contributions

**2. Preliminaries** (1 page)
- Flow matching + π₀ architecture
- On-policy gradient vanishing 分析

**3. Q-Guided Flow Matching** (2.5 pages)
- 3.1 Target perturbation + Theorem 1 (non-vanishing)
- 3.2 Q-network design + TD learning
- 3.3 Algorithm 1
- 3.4 Implementation details ($\eta$ schedule, warm-up, multi-epoch)

**4. Theoretical Analysis** (1.5 pages)
- Theorem 2: Implicit policy gradient
- Theorem 3: Improvement guarantee
- Theorem 4: Convergence rate
- Theorem 5: Unification (PA-RL, AWM, πRL as special cases)

**5. Experiments** (2.5 pages)
- 5.1 Main results (LIBERO, ManiSkill)
- 5.2 Training curves + convergence speed comparison
- 5.3 Ablations ($\eta$, warm-up, epochs, Q architecture)
- 5.4 Gradient norm analysis (three carriers)

**6. Related Work** (0.5 pages)

**7. Conclusion** (0.5 pages)

**Appendix**:
- A: Full proofs
- B: Extended experiments
- C: Hyperparameter sensitivity
- D: Comparison with PA-RL

---

## 九、时间线

| 时间 | 里程碑 |
|------|--------|
| 2026.04 W1-2 | 在 πRL (RLinf) codebase 上实现 QGFM（~200 行改动）|
| 2026.04 W3-4 | LIBERO-Spatial/Object 快速验证：Q 收敛 + success rate 上涨 |
| 2026.05 W1-2 | LIBERO-Long + Goal：temporal credit 验证 |
| 2026.05 W3-4 | ManiSkill 大规模验证 |
| 2026.06 W1-2 | 消融实验 + gradient analysis |
| 2026.06 W3-4 | 理论推导 + 论文写作 |
| 2026.07 W1-2 | 内部 review + 修改 |
| 2026.07 底 | 提交 NeurIPS 2026 |

**硬件**：32×A100。LIBERO 用 8×A100（64 并行环境），ManiSkill 用 24×A100（320 并行环境）。和 πRL 的 setting 完全一致。

---

## 十、风险评估

| 风险 | 等级 | Mitigation |
|------|------|-----------|
| Q 不准导致 $\nabla_a Q$ 方向错 | **中** | $\eta$ warm-up + 归一化 + Double Q |
| 性能不如 πRL PPO | **中** | 即使稍差，simplicity + off-policy + 无 SDE 是 selling point |
| Reviewer 说"就是 PA-RL" | **中高** | 数学推导（Theorem 1-5）+ 消融（one-step vs multi-step）+ iterative vs single-round |
| $\eta$ 敏感 | **低** | 消融展示 robustness |
| Off-policy Q divergence | **低** | Double Q + target network + bounded action space |

---

## 十一、一句话总结

> **Flow VLA RL 的三类 gradient carrier 各有致命缺陷：CFM 误差 on-policy vanish，SDE noise 需要复杂的 ratio + noise schedule，Q-gradient 穿 ODE 不稳定。QGFM 提出第四种方式——用 Q-gradient 扰动 flow matching 的 target action，不穿过 ODE、不需要 SDE、不需要 importance ratio，只需标准 flow matching + Q-learning，在 π₀/π₀.₅ 上达到 competitive performance。**
