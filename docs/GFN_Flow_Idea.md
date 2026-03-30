# GFN-Flow: GFlowNet-Guided Denoising for Diversity-Preserving RL of Flow-based VLAs

> **一句话总结**：将GFlowNet的reward-matching范式引入flow-based VLA的RL训练——把K步去噪过程建模为GFlowNet的DAG，用trajectory balance condition替代policy gradient，实现diversity-preserving的策略改进。不需要似然ratio，不需要环境level的critic，天然保持策略多样性。

---

## 1. 动机：为什么需要一个全新的范式

### 1.1 现有Flow+RL方法的共同局限

截至目前，所有flow-based policy的RL方法都属于**reward maximization**范式：

| 类别 | 代表方法 | 核心机制 |
|------|---------|---------|
| On-policy PG | πRL, ReinFlow, FPO, RLFM | 算似然(或代理) → PPO/GRPO |
| Off-policy AC | FlowRL(ByteDance), SAC Flow, FQL | Q-network → actor-critic |
| Noise-space RL | DSRL | 冻结权重，RL优化噪声输入 |
| Contrastive | DiffusionNFT, π-StepNFT | 正负镜像构造 → 对比loss |

**共同目标**：$\max_\theta \mathbb{E}_{\pi_\theta}[R]$

**共同后果**：策略收敛到单一最优mode → **mode collapse** → 面对OOD或不同初始条件时fragile。

### 1.2 Mode Collapse在VLA中的具体表现

**证据1：OOD泛化差距**

πRL(PPO)在ManiSkill上：IND 78.8% → OOD Vision 61.1%（下降17.7%）。PPO收敛到依赖特定视觉细节的单一策略，背景变化后该策略失效。

**证据2：Individual task波动**

πRL(PPO)在LIBERO-Long平均90.2%，但individual task间方差大。PPO倾向于找到一种策略后collapse到该策略，如果该策略恰好在某些初始条件下不适用就失败。

**证据3：Action chunk的多模态性被浪费**

Flow matching本身能建模multimodal action distribution。但reward maximization训练倾向于将所有概率质量推向单一mode，浪费了flow model的表达能力。

### 1.3 GFlowNet：一个根本不同的范式

GFlowNet不是maximize expected reward，而是学一个生成策略使得：

$$\pi(x) \propto R(x)$$

即生成概率正比于reward。这意味着：
- 高reward的行为被更频繁地生成 → 性能好
- 低reward但正的行为不会被完全消灭 → **天然保持diversity**
- 不会mode collapse到单一策略 → **OOD更robust**

### 1.4 为什么Flow VLA + GFlowNet是天然的结合

Flow VLA的K步去噪过程 **天然构成GFlowNet的DAG结构**：

- Source节点：初始噪声 $A^0$
- 中间节点：去噪中间状态 $A^{\tau_1}, A^{\tau_2}, A^{\tau_3}$
- Sink节点：最终动作 $A^1$
- 边：velocity field驱动的去噪转移

更关键的是：flow matching的**forward process（加噪）**天然提供了GFlowNet所需的**backward policy**——零额外参数。

这使得GFlowNet在flow VLA上的实现比在一般continuous domain上更简洁。

---

## 2. 背景

### 2.1 Flow-based VLA的动作生成

π0/π0.5的action generation：VLM提取多模态特征 → KV-cache传给Action Expert → K=4步ODE/SDE去噪：

$$A^0 \sim \mathcal{N}(0, I) \xrightarrow{v_\theta(\cdot, \tau_1, s)} A^{\tau_1} \xrightarrow{v_\theta(\cdot, \tau_2, s)} A^{\tau_2} \xrightarrow{v_\theta(\cdot, \tau_3, s)} A^{\tau_3} \xrightarrow{v_\theta(\cdot, \tau_4, s)} A^1 = a$$

ODE更新：$A^{\tau_{k+1}} = A^{\tau_k} + v_\theta(A^{\tau_k}, \tau_k, s) \cdot \delta_k$

条件概率路径：$A^\tau = \tau a_0 + (1-\tau)\epsilon$，目标velocity $u = a_0 - \epsilon$

SFT loss：$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{\tau, a_0, \epsilon}[\|v_\theta(A^\tau, \tau, s) - u\|^2]$

### 2.2 GFlowNet基础

**GFlowNet**学习一个在DAG $\mathcal{G} = (\mathcal{S}, \mathcal{A})$ 上的generative policy，使终端状态的生成概率正比于给定的reward函数。

**Trajectory Balance (TB) Condition**：对完整trajectory $\tau = (s_0 \to s_1 \to ... \to s_n)$：

$$Z \cdot \prod_{k=0}^{n-1} P_F(s_{k+1}|s_k) = R(s_n) \cdot \prod_{k=0}^{n-1} P_B(s_k|s_{k+1})$$

取log平方得TB loss：

$$\mathcal{L}_{TB} = \left(\log Z + \sum_k \log P_F(s_{k+1}|s_k) - \log R(s_n) - \sum_k \log P_B(s_k|s_{k+1})\right)^2$$

**Sub-Trajectory Balance (SubTB)**：引入state flow function $F(s)$，对任意子轨迹$(s_j \to ... \to s_k)$：

$$\mathcal{L}_{SubTB} = \left(\log F(s_j) + \sum_{i=j}^{k-1} \log P_F(s_{i+1}|s_i) - \log F(s_k) - \sum_{i=j}^{k-1} \log P_B(s_i|s_{i+1})\right)^2$$

边界条件：$F(s_n) = R(s_n)$（终端节点的flow等于reward）。

SubTB优势：更细粒度的信用分配、更稳定的训练、不需要完整trajectory。

### 2.3 Continuous GFlowNet

GFlowNet已被扩展到continuous state spaces：

- Lahlou et al. 2023："A Theory of Continuous Generative Flow Networks" → 理论基础
- Zhang et al. 2023："Diffusion GFlowNet" → diffusion作为GFlowNet的forward policy
- Sendera et al. 2024："Improved Off-policy Training of Diffusion Samplers"

这些工作验证了continuous GFlowNet的可行性。我们的setting比图像生成更简单：DAG只有4步，action space是finite-dimensional（7-20维）。

---

## 3. 方法：GFN-Flow

### 3.1 将去噪过程建模为GFlowNet DAG

对每个环境步（给定状态$s$），去噪过程定义了一个K=4步的DAG：

```
Source: A^0 ~ N(0, I)
  │
  ├── τ=τ_1 ──→ A^{τ_1}
  │                │
  │                ├── τ=τ_2 ──→ A^{τ_2}
  │                │                │
  │                │                ├── τ=τ_3 ──→ A^{τ_3}
  │                │                │                │
  │                │                │                ├── τ=τ_4 ──→ A^1 = a  (Sink)
```

**DAG的元素定义**：

| GFlowNet概念 | Flow VLA对应物 |
|-------------|--------------|
| State $s_k$ | 去噪中间状态 $(A^{\tau_k}, \tau_k)$ |
| Source | $(A^0, \tau=0)$，$A^0 \sim \mathcal{N}(0, I)$ |
| Sink | $(A^1, \tau=1)$，即执行的动作 $a$ |
| Forward policy $P_F$ | SDE去噪转移 |
| Backward policy $P_B$ | 加噪过程（flow matching forward process） |
| Terminal reward $R$ | 环境奖励 $R(a, s)$ |
| State flow $F$ | 去噪中间状态的"潜力"函数 |

### 3.2 Forward Policy

引入噪声使去噪过程stochastic（GFlowNet需要stochastic policy）：

$$P_F(A^{\tau_{k+1}} | A^{\tau_k}, s) = \mathcal{N}\left(\mu_k,\; \sigma_k^2 I\right)$$

其中均值由velocity field决定：

$$\mu_k = A^{\tau_k} + v_\theta(A^{\tau_k}, \tau_k, s) \cdot \delta_k$$

$\sigma_k$可以是固定的或可学习的。

**和πRL(Flow-SDE)的形式区别**：SDE formulation相似，但训练目标完全不同——πRL用$P_F$算似然ratio做PPO；GFN-Flow用$P_F$满足trajectory balance。

**Forward log-probability**（解析解）：

$$\log P_F(A^{\tau_{k+1}} | A^{\tau_k}, s) = -\frac{d}{2}\log(2\pi\sigma_k^2) - \frac{\|A^{\tau_{k+1}} - \mu_k\|^2}{2\sigma_k^2}$$

### 3.3 Backward Policy

GFlowNet需要backward policy $P_B(A^{\tau_k} | A^{\tau_{k+1}})$。Flow matching的**加噪过程**天然提供了这个：

在rectified flow中，条件概率路径：$A^\tau = \tau a_0 + (1-\tau)\epsilon$

给定$A^{\tau_{k+1}}$，反推$A^{\tau_k}$的分布：

$$P_B(A^{\tau_k} | A^{\tau_{k+1}}, s) = \mathcal{N}\left(\frac{\tau_k}{\tau_{k+1}} A^{\tau_{k+1}},\; \sigma_B^2(\tau_k, \tau_{k+1}) I\right)$$

其中 $\sigma_B^2$ 由条件概率路径的方差决定。

**关键优势：$P_B$是固定的，不需要学习。** 这大幅简化了训练——只需要学$v_\theta$（通过$P_F$）和$F_\psi$。

**Backward log-probability**：同样是解析高斯，精确可算。

### 3.4 State Flow Function

SubTB需要state flow function $F(A^{\tau_k}, \tau_k, s)$：

- 输入：去噪中间状态$A^{\tau_k}$、时间步$\tau_k$、环境状态$s$的compressed representation
- 输出：标量，表示从$(A^{\tau_k}, \tau_k)$出发到达高reward终端的"潜力"
- 参数化：小型MLP网络 $F_\psi$（~5M参数）
- 边界条件：$F(A^1, \tau=1, s) = R(a, s)$

**$F$的本质**：去噪过程内部的value function。但比环境level的critic简单得多——只需要预测4步去噪以内的"流"，而非50-100步环境交互的cumulative reward。

**$F$的独特优势**：
- 不直接以原始视觉特征为输入（用VLM的compressed token），不易过拟合视觉细节
- 4步DAG使得$F$的学习任务极其简单
- 和πRL的critic不同，$F$不预测绝对value，只需要满足相对的balance condition → 更容易学

### 3.5 训练目标：Sub-Trajectory Balance

对一条完整的去噪轨迹 $\tau = (A^0, A^{\tau_1}, A^{\tau_2}, A^{\tau_3}, A^1)$：

**Full TB Loss**：

$$\mathcal{L}_{TB} = \left(\log Z_\psi(s) + \log p(A^0) + \sum_{k=0}^{K-1} \log P_F(A^{\tau_{k+1}}|A^{\tau_k}, s) - \log R(A^1, s) - \sum_{k=0}^{K-1} \log P_B(A^{\tau_k}|A^{\tau_{k+1}}, s)\right)^2$$

**SubTB Loss**（更稳定，推荐使用）：

对每对 $(j, k)$（$0 \leq j < k \leq K$）：

$$\mathcal{L}_{SubTB}^{(j,k)} = \left(\log F_\psi(A^{\tau_j}, \tau_j, s) + \sum_{i=j}^{k-1} \log P_F(A^{\tau_{i+1}}|A^{\tau_i}, s) - \log F_\psi(A^{\tau_k}, \tau_k, s) - \sum_{i=j}^{k-1} \log P_B(A^{\tau_i}|A^{\tau_{i+1}}, s)\right)^2$$

总SubTB loss对所有子轨迹对取平均：

$$\mathcal{L}_{GFN} = \frac{1}{|\mathcal{P}|}\sum_{(j,k) \in \mathcal{P}} \mathcal{L}_{SubTB}^{(j,k)}$$

$K=4$时，共有$\binom{5}{2} = 10$个子轨迹对。

**边界enforcement**：

$$\mathcal{L}_{boundary} = \left(\log F_\psi(A^1, 1, s) - \log R(A^1, s)\right)^2$$

### 3.6 Reward设计

GFlowNet要求 $R > 0$。处理方式：

**Binary reward (LIBERO)**：

$$R(a, s) = \begin{cases} 1.0 & \text{if episode succeeds} \\ \epsilon & \text{if episode fails} \end{cases}$$

$\epsilon$ 取较小正值（如0.01）。失败trajectory的flow不为零但很小 → 模型学到这些行为是unlikely但不impossible → 提供微弱负信号，同时保持numerical stability。

**Shaped reward (ManiSkill)**：

$$R(a, s) = \epsilon + (1-\epsilon) \cdot r_{\text{env}}$$

其中$r_{\text{env}} \in \{0, 0.1, 1.0\}$（0.1 for grasp, 1.0 for place）。

### 3.7 SFT正则化

防止catastrophic forgetting：

$$\mathcal{L}_{reg} = \mathbb{E}_{(s', a') \sim \mathcal{D}_{\text{SFT}}}\left[\|v_\theta(x_t, t, s') - (a' - \epsilon)\|^2\right]$$

**总训练目标**：

$$\mathcal{L}_{total} = \mathcal{L}_{GFN} + \alpha \cdot \mathcal{L}_{boundary} + \lambda \cdot \mathcal{L}_{reg}$$

---

## 4. 算法

```
Algorithm: GFN-Flow

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  - SFT-initialized flow VLA:  v_θ (velocity field, ~300M params)
  - State flow network:        F_ψ (small MLP, ~5M params)
  - SFT data:                  D_sft
  - Noise levels:              {σ_k}_{k=0}^{K-1} (fixed or learnable)
  - Environments
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each iteration m = 1, 2, ..., M:

  ┌─────────────────────────────────────────────────┐
  │ Phase 1: Rollout with Stochastic Denoising      │
  └─────────────────────────────────────────────────┘
  
  D_rollout ← ∅
  
  For each task (state s_0, language instruction c):
    For each env step i = 0, ..., H-1:
      
      // K-step stochastic denoising
      Sample A_i^0 ~ N(0, I)
      For k = 0 to K-1:
        μ_k = A_i^{τ_k} + v_θ(A_i^{τ_k}, τ_k, s_i) · δ_k
        A_i^{τ_{k+1}} ~ N(μ_k, σ_k² I)
        Store denoising step: (A_i^{τ_k}, A_i^{τ_{k+1}}, τ_k, s_i)
      
      // Execute action
      a_i = A_i^1
      Execute a_i → s_{i+1}
    
    // Get episode reward
    R_episode ∈ {0, 1} (or shaped)
    R_i = ε + (1-ε) · R_episode    // ensure R > 0
    
    // Store all denoising trajectories with reward
    D_rollout ← D_rollout ∪ {(s_i, {A_i^{τ_k}}_{k=0}^K, R_i)}_{i=0}^{H-1}

  ┌─────────────────────────────────────────────────┐
  │ Phase 2: SubTB Loss Computation                  │
  └─────────────────────────────────────────────────┘
  
  For each denoising trajectory (s, {A^{τ_k}}_{k=0}^K, R) in batch:
    
    // Forward log-probs (analytic Gaussian)
    For k = 0 to K-1:
      μ_k = A^{τ_k} + v_θ(A^{τ_k}, τ_k, s) · δ_k
      log_PF_k = -d/(2)·log(2πσ_k²) - ||A^{τ_{k+1}} - μ_k||² / (2σ_k²)
    
    // Backward log-probs (fixed, analytic Gaussian)
    For k = 0 to K-1:
      μ_B = (τ_k / τ_{k+1}) · A^{τ_{k+1}}
      log_PB_k = -d/(2)·log(2πσ_B²) - ||A^{τ_k} - μ_B||² / (2σ_B²)
    
    // State flow predictions
    For k = 0 to K:
      f_k = F_ψ(A^{τ_k}, τ_k, s)    // scalar output
    
    // SubTB loss over all sub-trajectory pairs
    L_GFN = 0
    For j = 0 to K-1:
      For k = j+1 to K:
        residual = log(f_j) + Σ_{i=j}^{k-1} log_PF_i
                   - log(f_k) - Σ_{i=j}^{k-1} log_PB_i
        L_GFN += residual²
    L_GFN /= |pairs|
    
    // Boundary condition
    L_boundary = (log(f_K) - log(R))²

  ┌─────────────────────────────────────────────────┐
  │ Phase 3: SFT Regularization                      │
  └─────────────────────────────────────────────────┘
  
  Sample (s', a') from D_sft
  Sample t ~ U[0,1], ε ~ N(0,I)
  x_t = (1-t)·a' + t·ε
  L_reg = ||v_θ(x_t, t, s') - (a' - ε)||²

  ┌─────────────────────────────────────────────────┐
  │ Phase 4: Update                                  │
  └─────────────────────────────────────────────────┘
  
  L_total = L_GFN + α · L_boundary + λ · L_reg
  
  Update θ (velocity field)     via ∇_θ L_total
  Update ψ (state flow network) via ∇_ψ L_total
  Optionally update σ_k         via ∇_{σ_k} L_total
  
  // Soft update rollout policy
  θ_old ← β · θ_old + (1-β) · θ

Output: Optimized v_θ, F_ψ
```

---

## 5. 理论分析

### Theorem 1: Flow Matching — GFlowNet Duality

**Statement**：Flow VLA的K步SDE去噪过程定义了一个continuous GFlowNet的DAG。当trajectory balance condition精确满足时，终端动作的生成分布满足：

$$\pi_\theta(a|s) \propto R(a, s)$$

**Proof sketch**：

定义DAG：states $\mathcal{S} = \{(A^\tau, \tau) : \tau \in \{\tau_0, ..., \tau_K\}\}$，transitions由$P_F$定义。

TB condition：$Z(s) \cdot p(A^0) \cdot \prod_k P_F(A^{\tau_{k+1}}|A^{\tau_k}) = R(A^1, s) \cdot \prod_k P_B(A^{\tau_k}|A^{\tau_{k+1}})$

对$A^0$积分（marginalizing out intermediate states）：

$$Z(s) \cdot \int \prod_k P_F \cdot p(A^0) \, dA^0 \cdots dA^{\tau_{K-1}} = R(A^1, s) \cdot \int \prod_k P_B \, dA^{\tau_1} \cdots dA^{\tau_{K-1}}$$

左侧积分 = $\pi_\theta(A^1|s)$（marginal生成概率）。右侧积分 = $P_B(A^0|A^1) \cdot \text{const}$。

因此 $\pi_\theta(A^1|s) = R(A^1, s) / Z(s)$。$\square$

### Theorem 2: Diversity Guarantee

**Statement**：当TB condition满足且$R(a, s) > 0$ $\forall a \in \text{support}$时，$\pi_\theta$不会mode collapse——所有positive-reward actions保持非零生成概率。

**Proof**：由$\pi_\theta(a|s) = R(a, s) / Z(s)$，$R > 0 \Rightarrow \pi_\theta > 0$。$\square$

**对比**：reward maximization的PPO可以将次优mode的概率推到0。GFN-Flow不会。

### Theorem 3: Credit Assignment via State Flow

**Statement**：SubTB的梯度对第$k$步去噪的贡献正比于$|\nabla_{A^{\tau_k}} \log F_\psi|$。$F$变化大的步（关键决策步）自动获得更大的训练信号。

**Proof sketch**：SubTB residual中包含$\log F_\psi(A^{\tau_k})$项。对$\theta$求导通过chain rule：

$$\frac{\partial \mathcal{L}_{SubTB}}{\partial \theta} \propto \text{residual} \cdot \frac{\partial \log P_F}{\partial \theta}$$

$\log P_F$对$\theta$的梯度在velocity prediction误差大的步（$F$变化剧烈的步）更大。$\square$

### Theorem 4: Connection to Maximum Entropy RL

**Statement**：GFlowNet的reward-matching目标 $\pi \propto R$ 等价于以下maximum entropy RL问题的解：

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi[\log R(a, s)] + H(\pi)$$

其中$H(\pi) = -\mathbb{E}_\pi[\log \pi]$是策略熵。

**Proof**：Lagrangian对偶。带约束$\int \pi(a|s) da = 1$的优化：

$$\mathcal{L} = \mathbb{E}_\pi[\log R] + H(\pi) + \lambda(\int \pi \, da - 1)$$

对$\pi$求变分导数令其为零：$\log R(a,s) - \log\pi(a|s) - 1 + \lambda = 0$

解得 $\pi(a|s) \propto R(a,s)$。$\square$

**意义**：GFN-Flow隐式实现了entropy-regularized RL，但不需要显式计算策略熵（对flow policy是intractable的）。这是相对于直接用SAC的关键优势。

---

## 6. 与现有方法的全面对比

### 6.1 Paradigm层面

| 维度 | Policy Gradient方法 | Contrastive方法 | Noise-space方法 | **GFN-Flow** |
|------|-------------------|----------------|----------------|-------------|
| 代表 | πRL, ReinFlow, FPO | DiffusionNFT, π-StepNFT | DSRL | **本文** |
| 优化目标 | $\max \mathbb{E}[R]$ | $\max \mathbb{E}[R]$ (隐式) | $\max \mathbb{E}[R]$ | **$\pi \propto R$** |
| Mode collapse | 会 | 会 | 会（capacity内） | **不会** |
| 需要似然 | ✅ | ❌ | ❌ | **❌** |
| 需要环境Critic | ✅ (PPO) / ❌ (GRPO) | ❌ | ✅ (noise-space) | **❌** |
| 信用分配 | GAE (需Critic) | Episode-level | Noise-space critic | **State flow $F$** |
| 额外网络 | Critic (+noise net) | 无 | Noise policy + critic | **State flow $F_\psi$** (~5M) |
| 改权重 | ✅ | ✅ | ❌ | **✅** |
| Diversity | 低 | 低 | 中 | **高** |

### 6.2 具体方法对比

#### vs πRL (PPO/GRPO)

- πRL需要算似然（Flow-Noise或Flow-SDE）→ GFN-Flow不需要
- πRL的PPO需要环境level critic → GFN-Flow只需去噪level的state flow
- πRL的critic以full visual features为输入 → 过拟合 → OOD差
- GFN-Flow的$F_\psi$只看compressed tokens + 去噪中间状态 → 更不易过拟合

#### vs DiffusionNFT / π-StepNFT

- NFT构造人工镜像$v^\pm$ → 不一定realistic
- GFN-Flow的"负信号"来自trajectory balance的自然约束，不需要构造任何人工样本
- NFT本质是reward max（隐式）→ mode collapse；GFN-Flow是reward matching → 保持diversity

#### vs DSRL

- DSRL不改权重 → capacity受限
- GFN-Flow更新velocity field → 无capacity限制
- DSRL需要noise-space critic → GFN-Flow的$F_\psi$在更简单的空间（4步DAG内）

#### vs SAC (MaxEnt RL)

Theorem 4表明GFN-Flow ≈ MaxEnt RL。区别在于实现方式：
- SAC需要显式计算$H(\pi) = -\mathbb{E}[\log\pi]$ → 对flow policy intractable（需要Jacobian trace）
- GFN-Flow通过trajectory balance**隐式**实现entropy maximization → 不需要算$\log\pi$

### 6.3 核心优势总结

1. **全新范式**：GFlowNet从未被应用于robot policy learning。这不是现有算法的变体。
2. **Diversity-preserving**：理论保证不mode collapse（Theorem 2），对OOD泛化至关重要。
3. **无似然计算**：forward/backward log-prob都是解析高斯，精确可算，不需要Jacobian trace。
4. **无环境level critic**：State flow $F_\psi$只在4步去噪DAG内部运作，比环境MDP的value function简单得多，过拟合风险极低。
5. **自动信用分配**：SubTB的结构天然给去噪过程的不同步分配不同的训练信号（Theorem 3）。
6. **Backward policy免费**：flow matching的加噪过程天然提供$P_B$，零额外参数。

---

## 7. 实现细节

### 7.1 架构

- **Base model**：π0 (PaliGemma 3B VLM + ~300M flow action expert) 或 π0.5
- **RL阶段冻结VLM**，只更新action expert（与πRL一致）
- **State flow network** $F_\psi$：4-layer MLP，输入为concat(VLM compressed token, $A^{\tau_k}$, $\tau_k$)，输出为正标量（softplus activation）。~5M参数。
- **$P_B$**：固定，解析高斯（主实验）。Learned $P_B$作为ablation。

### 7.2 噪声level设计

$\sigma_k$的选择影响exploration和训练稳定性：

- $\sigma_k$太小 → 退化为ODE → $P_F$变成delta distribution → $\log P_F$趋于$-\infty$ → TB loss blow up
- $\sigma_k$太大 → 探索过度 → 生成动作偏离合理范围

建议：$\sigma_k \in [0.05, 0.3]$，或使用可学习的$\sigma_k$（per-step）。

### 7.3 关键超参数

| 超参数 | 含义 | 建议范围 | 默认值 |
|--------|------|---------|--------|
| $\epsilon$ | 失败trajectory的最小reward | [0.001, 0.1] | 0.01 |
| $\sigma_k$ | Forward policy噪声level | [0.05, 0.3] | 0.1 |
| $\alpha$ | Boundary loss权重 | [0.1, 10.0] | 1.0 |
| $\lambda$ | SFT正则权重 | [0.01, 1.0] | 0.1 |
| K | 去噪步数 | 3-8 | 4 |
| $\beta$ | Rollout policy EMA rate | [0.9, 0.999] | 0.95 |

### 7.4 计算开销

| 操作 | 相对于SFT的开销 |
|------|---------------|
| Stochastic rollout（和πRL SDE相同） | ~1x |
| Forward/backward log-prob计算 | 可忽略（解析解） |
| $F_\psi$ forward pass（10对子轨迹） | ~0.05x |
| SubTB loss计算 | ~0.05x |
| SFT正则 | ~0.3x |
| **总计** | **~1.4x** |

远低于πRL（~2x，需训练critic）和GRPO（~Nx，N条rollout）。

---

## 8. 实验设计

### 8.1 Benchmarks

| Benchmark | 任务 | 奖励 | 核心测试 |
|-----------|------|------|---------|
| LIBERO (4 suites) | 40个操控任务 | Binary (0/1) | 性能 + diversity |
| ManiSkill | 4352种pick-and-place组合 | Shaped (0/0.1/1.0) | IND + OOD泛化 |

### 8.2 Baselines

| 方法 | 类型 | 来源 |
|------|------|------|
| SFT (few-shot) | 纯监督 | 基线 |
| πRL (Flow-SDE + PPO) | PG + Critic | πRL论文 |
| πRL (Flow-SDE + GRPO) | PG + 组排序 | πRL论文 |
| DSRL | Noise-space RL | DSRL论文 |
| RFT (RWFM) | 成功样本SFT | RLFM论文 |
| **GFN-Flow (ours)** | **GFlowNet** | **本文** |

### 8.3 主实验与预期

#### LIBERO

| 方法 | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| SFT (π0) | 65.3 | 64.4 | 49.8 | 51.2 | 57.6 |
| πRL (PPO) | 98.4 | 99.4 | 96.2 | 90.2 | 96.1 |
| πRL (GRPO) | 97.8 | 97.8 | 83.2 | 81.4 | 90.0 |
| **GFN-Flow** | **~96** | **~98** | **~92** | **~88** | **~93.5** |

预期：接近πRL(PPO)。可能在Long上略低（没有GAE级别的信用分配），但per-task consistency更好（diversity保护）。

#### ManiSkill OOD

| 方法 | IND | OOD Vision | OOD Semantic | OOD Avg |
|------|-----|-----------|-------------|---------|
| πRL (PPO) | 78.8 | 61.1 | 25.4 | 39.3 |
| **GFN-Flow** | **~75** | **~68** | **~35** | **~47** |

预期：IND略低于PPO（reward matching vs reward max），但**OOD显著更好**——diversity意味着策略不依赖单一视觉模式，更robust。

### 8.4 Diversity量化实验

**实验设计**：对同一初始状态$s_0$，各方法采样$N=100$次动作，分析：

| 指标 | 计算方式 | 预期 |
|------|---------|------|
| Action Entropy | $H(\{a_1, ..., a_N\})$ via KDE | GFN-Flow >> πRL |
| Success Mode Count | DBSCAN clustering on successful actions | GFN-Flow >> πRL |
| Success Rate Variance | Per-task variance across multiple seeds | GFN-Flow << πRL |

**可视化**：2D t-SNE of action chunks，colored by success/failure。
- πRL：紧密聚集在单一cluster
- GFN-Flow：分散在多个clusters，且多数是green（成功）

### 8.5 消融实验

| 消融 | 目的 |
|------|------|
| TB vs SubTB | SubTB是否更稳定？ |
| Fixed vs learned $P_B$ | Backward policy的影响 |
| Fixed vs learned $\sigma_k$ | 噪声level的自适应性 |
| $\epsilon = 0.01$ vs $0.001$ vs $0.1$ | 失败trajectory的reward floor |
| 有/无SFT正则 | 防遗忘的必要性 |
| GFN-Flow vs MaxEnt (SAC-style entropy bonus) | GFlowNet vs 直接entropy regularization |
| 不同K（3,4,5,8） | DAG深度对训练的影响 |

### 8.6 分析实验

1. **State flow $F$的可视化**：$F(A^{\tau_k}, \tau_k, s)$在去噪过程中的变化。成功trajectory的$F$应该单调递增；失败trajectory的$F$应该在某步骤下降（识别错误决策步）。

2. **TB residual监控**：训练过程中TB residual趋向0 → balance condition被满足 → 生成分布收敛到$\propto R$。

3. **Generation diversity的演化**：训练过程中action entropy的变化曲线。πRL应该entropy单调下降（mode collapse）；GFN-Flow应该维持较高entropy。

4. **Per-task breakdown**：哪些任务GFN-Flow优于πRL（多模态任务）？哪些不如（单一最优策略的任务）？

---

## 9. 潜在风险与缓解

### 9.1 Continuous GFlowNet的训练稳定性

**风险**：GFlowNet在continuous space的训练经验有限，可能不稳定。

**缓解**：
- DAG只有4步，远比图像生成的100+步简单
- SubTB比TB更稳定（GFlowNet文献共识）
- Gradient clipping防止TB residual的平方导致梯度爆炸
- SFT正则提供额外的稳定性

### 9.2 Diversity在VLA benchmark上是否真的有用

**风险**：如果LIBERO/ManiSkill的最优策略是唯一的，diversity是负担不是优势。

**缓解**：
- 即使最优策略唯一，GFN-Flow的$\pi \propto R$仍会把大部分质量放在最优策略上（因为$R$最大），只是不会完全collapse
- OOD实验是diversity价值的直接验证
- 如果diversity确实不帮助IND，那OOD的提升就是核心卖点

### 9.3 和MaxEnt RL的区别是否显著

**审稿人可能的challenge**："Theorem 4说GFN ≈ MaxEnt RL，直接用SAC不就行了？"

**回应**：
1. SAC需要显式计算$\log\pi(a|s)$ → 对flow policy intractable → 这正是πRL花整篇论文解决的问题
2. GFN-Flow通过trajectory balance**隐式**实现entropy maximization，不需要算$\log\pi$
3. 两者虽然在理论极限下等价，但优化路径和实现方式完全不同
4. 消融实验直接对比GFN-Flow和SAC-style entropy bonus → 量化差异

### 9.4 $P_B$的近似质量

**风险**：Fixed $P_B$（加噪过程）可能不是最优的backward policy，导致训练效率低。

**缓解**：
- GFlowNet文献表明fixed $P_B$在short DAG上效果可接受
- Learned $P_B$作为ablation，如果差距大则升级为主方案
- 4步DAG的$P_B$影响有限（backward path很短）

### 9.5 State flow $F_\psi$的学习质量

**风险**：$F_\psi$在训练早期不准确，SubTB loss的梯度有噪声。

**缓解**：
- $F_\psi$的学习任务极其简单（4步DAG内预测）
- 可以用warm-up阶段先训练$F_\psi$（冻结$v_\theta$，只更新$\psi$）
- TB loss本身提供了$F_\psi$的监督信号，两者联合训练

---

## 10. 论文结构

### Title
"GFN-Flow: GFlowNet-Guided Denoising for Diversity-Preserving RL of Flow-based VLAs"

### Abstract
Flow-based VLAs的RL训练面临似然不可解和mode collapse两大难题。所有现有方法都属于reward maximization范式，训练后策略collapse到单一mode，导致OOD泛化差。我们提出GFN-Flow，首次将GFlowNet引入VLA的RL训练。通过将K步去噪过程建模为GFlowNet的DAG，我们学习reward-matching的策略——生成概率正比于reward。GFN-Flow不需要似然计算（forward/backward log-prob均为解析高斯），不需要环境level的critic（只需轻量的state flow function），且理论保证diversity不collapse。在LIBERO和ManiSkill上的实验表明，GFN-Flow在IND任务上达到competitive性能，在OOD泛化上显著优于reward maximization方法。

### 论文结构

1. **Introduction**：Flow VLA + RL的重要性 → 现有方法都是reward max → mode collapse导致OOD差 → 我们的范式转变

2. **Related Work**：Flow+RL methods综述（πRL, DSRL, FPO, NFT等） → GFlowNet背景 → Continuous GFlowNet

3. **Preliminaries**：Flow matching for VLA → GFlowNet (TB, SubTB)

4. **Method: GFN-Flow**
   - 4.1 去噪DAG formulation
   - 4.2 Forward policy (stochastic denoising)
   - 4.3 Backward policy (flow forward process)
   - 4.4 State flow function
   - 4.5 SubTB training objective
   - 4.6 SFT regularization

5. **Theoretical Analysis**
   - Flow-GFlowNet duality (Theorem 1)
   - Diversity guarantee (Theorem 2)
   - Credit assignment via state flow (Theorem 3)
   - Connection to MaxEnt RL (Theorem 4)

6. **Experiments**
   - 6.1 LIBERO main results
   - 6.2 ManiSkill main results + OOD
   - 6.3 Diversity quantification
   - 6.4 Ablations
   - 6.5 Analysis

7. **Conclusion + Future Work**

---

## 11. 代码改动估算

基于πRL开源代码（RLinf框架）：

| 模块 | 改动 | 行数估算 |
|------|------|---------|
| Rollout | 存储完整去噪轨迹 | ~20行 |
| State flow network | 新增小MLP | ~50行 |
| Forward/backward log-prob | 解析高斯计算 | ~30行 |
| SubTB loss | 子轨迹enumeration + loss计算 | ~60行 |
| Boundary loss | 终端条件 | ~10行 |
| SFT regularization | 复用现有CFM loss | ~10行 |
| Training loop | 替换PPO objective为GFN objective | ~40行 |
| **总计** | | **~220行核心代码** |

需要删除的模块：PPO objective, advantage estimation, critic network, 似然计算。

**净代码量变化**：可能减少，因为删除的比新增的多。

---

## 12. 48小时快速验证方案

### Step 1：验证SubTB训练是否收敛（Day 1上午）

在一个简单的2D continuous bandit上：
- 状态：固定
- 动作：2D
- Reward：bimodal Gaussian（两个高reward区域）
- 4步flow去噪
- 用SubTB训练
- 验证：生成分布是否覆盖两个modes？TB residual是否趋零？

### Step 2：验证在LIBERO-Spatial上的可行性（Day 1下午-Day 2）

- 用π0 SFT checkpoint
- 实现GFN-Flow training loop
- 跑100个iterations
- 检查：成功率是否从57.6%开始提升？训练是否稳定？

### Step 3：初步diversity对比（Day 2）

- 同一状态采样100次动作
- 对比GFN-Flow vs SFT vs PPO（如果πRL checkpoint可用）
- 画action分布图

如果Step 1失败 → continuous GFlowNet不可行，需要换方向
如果Step 2不提升 → 超参数问题或formulation问题，需要调试
如果Step 2提升但Step 3无diversity → diversity claim不成立，需要重新定位
