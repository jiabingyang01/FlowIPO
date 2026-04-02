# Deviation-Enhanced Contrastive Optimization for Flow-based VLA

## DECO：用参考偏差实现无Critic步级信用分配

---

## 一、Problem Statement

### 1.1 当前最优方案及其瓶颈

π-StepNFT（GigaAI, 2026.03）是当前**无Critic无似然**路线的最佳实践。其核心流程：

1. 将flow matching的ODE转换为SDE（拓宽探索空间）
2. 对每个SDE去噪步构造镜像分支（正向v⁺/反向v⁻）
3. 用episode级二值标签 $y = 2r - 1$ 做logistic对比排序损失
4. 直接优化向量场，不需要 $\log\pi$ 也不需要Critic

π-StepNFT在ManiSkill OOD上全面碾压PPO（+11.1%），因为省去了Critic就避免了视觉纹理过拟合。但存在一个**核心瓶颈**：

- **长horizon任务被PPO拉开**：LIBERO-Long上 π₀: 86.7% vs πRL(PPO) 90.2%；π₀.₅: 79.8% vs 93.0%

### 1.2 瓶颈根因分析

瓶颈可追溯到一个**单一根因**：

**根因：Episode级标签导致零时间信用分配。** π-StepNFT用标量 $y = 2r-1$ 作为所有环境步的统一学习信号。一条200步轨迹中，第5步和第195步拿到完全相同的 $y$ 值。

这导致三个级联问题：

**问题A：关键步与无关步等权学习。** 假设成功轨迹中第50步做了关键抓取，而第1-49步只是常规接近。π-StepNFT对所有步给出 $y=+1$，第1步（几乎无关）和第50步（决定成败）获得完全相同的梯度权重。

**问题B：错误归因。** 失败轨迹中前180步可能全是正确操作，只有第181步出了致命错误。但 $y=-1$ 被均匀分配给所有步，前180步的正确行为也被惩罚。

**问题C：长horizon放大效应。** 轨迹越长，"关键步/总步数"的比例越低，信噪比越差。这直接解释了π-StepNFT在Long任务上的性能下降。

本工作聚焦于**在不引入Critic的前提下解决这个根因**——因为引入Critic会丧失π-StepNFT最大的优势（OOD泛化）。

---

## 二、Background：步级信用分配的现有方案

### 2.1 为什么不能直接加Critic

PPO通过learned value function $V_\phi(s)$ 做GAE，提供步级优势 $\hat{A}_t$。但πRL论文（Tab.3 ManiSkill OOD）的实验直接暴露了代价：

| 方法 | IND | Vision OOD | Semantic OOD | OOD Avg. |
|------|-----|-----------|-------------|----------|
| πRL (PPO) | 78.8 | 61.1 | 25.4 | 39.3 |
| π-StepNFT | 79.2 | 69.1 | 49.1 | **50.4** |

Critic从视觉-语言嵌入估计价值，过拟合到训练分布的视觉纹理和语言表述。**加Critic换来了IND的步级信用分配，但付出了OOD 11.1%的代价。**

### 2.2 其他无Critic步级方案的局限

| 方案 | 代表方法 | 局限 |
|------|---------|------|
| LLM生成密集奖励 | TGRPO | 需要LLM前向传播，额外计算开销大，奖励质量依赖LLM能力，不适用于flow VLA |
| 组相对优势 | SimpleVLA-RL (GRPO) | 需要同状态多次采样，但flow VLA的同观测多次rollout开销极大 |
| 世界模型进度预测 | SC-VLA | 需要联合训练世界模型模块，增加架构复杂度 |
| 离线学习成功概率 | π-StepNFT Sec.5提到的扩展 | 需要额外的监督数据和预训练，论文自身未实现 |

**核心矛盾**：所有现有的步级信用分配方案都需要引入**额外组件**，而π-StepNFT的OOD优势恰恰来自"什么额外组件都不加"。

### 2.3 参考偏差的信息论直觉

策略优化理论中的Natural Policy Gradient告诉我们：策略更新的有效性与策略变化量正相关。如果当前策略在某状态下的行为与参考策略几乎一样，该状态对策略改进的贡献接近零。

**核心观察**：

> **策略偏离参考模型的程度本身就是一个免费的步级信息源。** 不需要任何额外网络——只需要一次冻结参考模型的前向传播，就能知道当前策略在哪些步做出了"不一样的决策"。结合episode结果，就能推断"不一样的决策"是好是坏。

直觉：如果一条成功轨迹中，策略在第50步大幅偏离了参考模型（而其他步基本没变），那大概率是第50步的新行为导致了成功——应该重点强化。反之亦然。

这和RL中KL penalty的哲学一脉相承——PPO/TRPO用 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 来约束策略更新幅度，本质上也是在用参考策略作为信息锚点。DECO只是将这个信息**从全局正则化转化为步级权重**。

---

## 三、方法：DECO

### 3.1 Setup

- **预训练策略**：$\pi_{\theta_{\text{ref}}}$ = flow matching VLA（如π₀），冻结不动，作为参考锚点
- **Rollout策略**：$\pi_{\theta_{\text{old}}}$，EMA更新
- **当前策略**：$\pi_\theta$，梯度下降更新
- **环境**：MDP $(S, A, P, R)$，binary reward（成功1/失败0）
- **Action chunking**：每步生成C维action chunk

### 3.2 参考偏差计算（唯一的新组件）

对rollout收集的每个环境步 $i$（状态 $s_i$），计算当前rollout策略与冻结参考策略在flow空间中的行为差异：

$$D_i = \frac{1}{K}\sum_{j=0}^{K-1}\left\|v_{\theta_{\text{ref}}}(x_{t_j}, t_j, s_i) - v_{\theta_{\text{old}}}(x_{t_j}, t_j, s_i)\right\|^2$$

其中 $\{x_{t_j}\}_{j=0}^{K-1}$ 是rollout时SDE采样经过的K个中间状态（已经记录在缓冲区中，不需要额外采样）。

**计算开销分析**：

- $v_{\theta_{\text{ref}}}$ 是冻结的，不需要梯度，用 `torch.no_grad()` + 半精度推理
- K通常取4（和π-StepNFT一致），即每个环境步4次额外action expert前向传播
- 可以和 $v_{\theta_{\text{old}}}$ 的前向传播**合并batch**一次完成，实际wall-clock额外开销约**20-25%**

### 3.3 步级标签构造（核心改动）

**Step 1 — Batch归一化：**

$$\bar{D} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} D_i, \quad \sigma_D = \text{std}(\{D_i\}_{i \in \mathcal{B}})$$

**Step 2 — Sigmoid映射：**

$$w_i = \sigma\!\left(\frac{D_i - \bar{D}}{\sigma_D}\right) \in (0, 1)$$

**Step 3 — 步级标签：**

$$\boxed{y_i = (2r - 1)\cdot\bigl[1 + \eta\cdot w_i\bigr]}$$

其中 $\eta > 0$ 是调制强度，唯一的新超参数。

**含义**：

| $D_i$ | r=1（成功） | r=0（失败） |
|-------|-----------|-----------|
| 大（策略偏离参考） | $y_i \gg +1$ → **重点强化** | $y_i \ll -1$ → **重点惩罚** |
| 小（策略未偏离） | $y_i \approx +1$ → 正常学 | $y_i \approx -1$ → 轻轻惩罚 |

**和π-StepNFT的关键区别**：

| | π-StepNFT | DECO（本方案） |
|---|---|---|
| 标签 | $y = 2r-1$（标量，episode级） | $y_i = (2r-1)(1+\eta w_i)$（向量，步级） |
| 信用分配 | 无（所有步等权） | 有（高偏差步获得更大权重） |
| 额外组件 | 无 | 一次冻结ref模型前向传播 |
| 额外超参 | 无 | $\eta$（一个标量） |
| $\eta=0$时 | — | **严格退化为π-StepNFT** |

### 3.4 DECO损失函数

对π-StepNFT的**唯一修改**——将标量 $y$ 替换为 $y_i$：

$$\ell_t^{\text{DECO}}(\theta) = \text{softplus}\!\left(\frac{y_i}{2}\cdot(E_{\theta,t}^+ - E_{\theta,t}^-)\right) + \lambda_{\text{TR}}\|\Delta v_\theta\|^2$$

其余所有组件完全不变：

- SDE rollout方式不变
- 镜像分支构造不变（$v_\theta^{\pm} = v_{\text{old}} \pm \beta\Delta v_\theta$）
- 方差归一化步误差 $E_{\theta,t}^{\pm}$ 不变
- 信任域正则 $\lambda_{\text{TR}}\|\Delta v_\theta\|^2$ 不变
- EMA更新策略不变

### 3.5 完整算法

---

#### Algorithm 1: DECO

**Input:** 预训练 flow VLA $\pi_{\theta_{\text{ref}}}$（冻结），信任域 $\beta$，SDE噪声 $\sigma$，偏差调制强度 $\eta$，正则系数 $\lambda_{\text{TR}}$，EMA衰减 $\alpha_m$

**Initialize:** $\theta \leftarrow \theta_{\text{ref}}$，$\theta_{\text{old}} \leftarrow \theta_{\text{ref}}$

**For** iteration $k = 1, 2, \ldots$ **do:**

**Phase 1 (SDE Rollout + 偏差记录):** 用 $\pi_{\theta_{\text{old}}}$ 在并行环境中执行 $G$ 条轨迹。

对每条轨迹的每个环境步 $i$：

1. 运行K步Flow-SDE solver，生成denoising链 $\{x_{t_j}\}_{j=0}^{K}$
2. 均匀采样一个solver步 $j \sim \mathcal{U}\{0,\ldots,K-1\}$，记录 $(x_t, x_{t^-}, v_{\text{old}}, t, s_i)$
3. **【新增】** 计算参考偏差（`torch.no_grad()`下）：

$$D_i = \frac{1}{K}\sum_{j=0}^{K-1}\left\|v_{\theta_{\text{ref}}}(x_{t_j}, t_j, s_i) - v_{\theta_{\text{old}}}(x_{t_j}, t_j, s_i)\right\|^2$$

4. 执行最终动作，收集环境反馈

Episode结束后获得终端信号 $r \in \{0, 1\}$。所有 $(x_t, x_{t^-}, v_{\text{old}}, t, s_i, D_i, r)$ 存入缓冲区 $\mathcal{D}$。

**Phase 2 (步级标签计算):**

$$\bar{D} = \text{mean}(\{D_i\}_{i \in \mathcal{D}}), \quad \sigma_D = \text{std}(\{D_i\}_{i \in \mathcal{D}})$$

$$y_i = (2r - 1)\cdot\left[1 + \eta\cdot\sigma\!\left(\frac{D_i - \bar{D}}{\sigma_D}\right)\right]$$

**Phase 3 (策略优化):** 对 mini-batch $\mathcal{B} \subset \mathcal{D}$：

$$v_\theta \leftarrow \pi_\theta(c, s, x_t, t)$$

$$\Delta v_\theta \leftarrow v_\theta - v_{\text{old}}$$

$$v_\theta^{\pm} \leftarrow v_{\text{old}} \pm \beta\cdot\Delta v_\theta$$

$$\mu_{\theta,t}^{\pm}, \Sigma_t \leftarrow \text{从SDE仿射结构计算}$$

$$E_{\theta,t}^{\pm} \leftarrow \|x_{t^-} - \mu_{\theta,t}^{\pm}\|_{\Sigma_t^{-1/2}}$$

$$\mathcal{L} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}}\left[\text{softplus}\!\left(\frac{y_i}{2}\cdot(E_{\theta,t}^+ - E_{\theta,t}^-)\right) + \lambda_{\text{TR}}\|\Delta v_\theta\|^2\right]$$

$$\theta \leftarrow \theta - \alpha\nabla_\theta \mathcal{L}$$

**Phase 4 (EMA更新):**

$$\theta_{\text{old}} \leftarrow \alpha_m\cdot\theta_{\text{old}} + (1-\alpha_m)\cdot\theta$$

清空缓冲区，回到Phase 1。

**End For**

**Return** $\pi_\theta$

---

### 3.6 实现：和π-StepNFT的代码差异

假设现有codebase是π-StepNFT。DECO只需要改动标签计算部分：

**π-StepNFT的标签**（需要改的部分）：
```python
# episode级二值标签
y = 2 * reward - 1  # 标量，广播到所有步
loss = softplus(0.5 * y * (E_plus - E_minus))
```

**DECO的标签**（替换为）：
```python
# === Phase 1: rollout时额外记录参考偏差 ===
# （在SDE rollout循环内，已有 v_old 和 x_t_all）
with torch.no_grad():
    v_ref = pi_ref(x_t_all, t_all, s_i)       # 冻结参考模型前向
    D_i = ((v_ref - v_old) ** 2).mean(dim=-1)  # 每步一个标量

# === Phase 2: 构造步级标签 ===
D_mean = D_buffer.mean()
D_std  = D_buffer.std() + 1e-8
w_i    = torch.sigmoid((D_i - D_mean) / D_std)
y_i    = (2 * reward - 1) * (1 + eta * w_i)   # 步级标签

# === Phase 3: 损失（只改了 y → y_i） ===
loss = softplus(0.5 * y_i * (E_plus - E_minus))
```

**改动量**：约15行代码。不需要新增网络、不需要改架构、不需要改rollout pipeline。

### 3.7 参考模型的选择

$\pi_{\theta_{\text{ref}}}$ 的自然选择是**SFT checkpoint**（Few-shot微调后、RL训练前的初始策略）。理由：

1. **语义一致**：SFT策略代表"RL训练前的行为基准"，$D_i$ 衡量的就是"RL让策略在这一步改变了多少"
2. **已经存在**：不需要额外训练或存储
3. **冻结稳定**：不随训练更新，避免信号漂移

**不选其他参考点的原因**：

| 候选参考 | 问题 |
|---------|------|
| $\pi_{\theta_{\text{old}}}$（当前rollout策略） | 每轮都变，偏差信号不稳定，失去"长期锚点"意义 |
| 随机初始化模型 | 偏差无语义，所有步偏差都大 |
| 训练中间checkpoint | 需要额外存储和选择逻辑 |

---

## 四、理论分析

### 4.1 梯度方向保持定理

**命题1.** DECO的梯度方向与π-StepNFT的oracle对齐性完全一致。

π-StepNFT的Theorem 4.4给出梯度形式：

$$-\nabla_\theta \ell_t(\theta) \propto \sigma(z_t)\cdot y \cdot \left(\frac{\partial v_\theta}{\partial \theta}\right)^\top B_t\,\Sigma_t^{-1}\,e_t$$

DECO中将 $y$ 替换为 $y_i = (2r-1)(1+\eta w_i)$。

**证明**：由于 $1 + \eta w_i > 0$ 恒成立（$w_i \in (0,1)$，$\eta > 0$），因此 $\text{sign}(y_i) = \text{sign}(y)$。

DECO梯度为：

$$-\nabla_\theta \ell_t^{\text{DECO}}(\theta) \propto \sigma(z_t^{\text{DECO}})\cdot |y_i| \cdot \text{sign}(y) \cdot \left(\frac{\partial v_\theta}{\partial \theta}\right)^\top B_t\,\Sigma_t^{-1}\,e_t$$

与π-StepNFT的梯度**方向完全一致**（都指向 $\text{sign}(y)\cdot\Sigma_t^{-1} e_t$），只是**幅度**被 $|y_i|$ 调制。

由于π-StepNFT的Theorem 4.4(c)证明了小步更新下梯度方向与oracle均值差 $\Delta\mu_t^*$ 对齐，DECO的梯度方向同样与 $\Delta\mu_t^*$ 对齐。$\square$

**直觉**：DECO不改变"往哪走"，只改变"每步走多快"——在关键步走快，在无关步走慢。

### 4.2 有界性

**命题2.** DECO的步级标签 $y_i$ 有界：$|y_i| \in [1,\; 1+\eta]$。

**证明**：$w_i = \sigma(\cdot) \in (0, 1)$，$|2r-1| = 1$，所以 $|y_i| = 1 + \eta w_i \in (1, 1+\eta)$。$\square$

**意义**：

1. 不存在梯度爆炸风险（$|y_i|$ 有上界）
2. 不存在梯度消失风险（$|y_i| > 1$，至少和baseline一样强）
3. $\eta$ 直接控制了最大/最小权重比：$\eta = 3$ → 最大 $|y_i| \approx 4$，最小 $|y_i| \approx 1$，比值 $\leq 4$

### 4.3 严格退化性

**命题3.** 当 $\eta = 0$ 时，DECO严格退化为π-StepNFT。

**证明**：$\eta = 0 \Rightarrow y_i = (2r-1)\cdot(1+0) = 2r-1 = y$。$\square$

**意义**：DECO是π-StepNFT的**严格超集**。在最坏情况下（偏差信号无信息量），调 $\eta = 0$ 即可回退到基线。**DECO不可能比π-StepNFT更差**（只要 $\eta$ 的sweep包含0）。

### 4.4 信噪比改善分析

**命题4.** 在"关键步假设"下，DECO的有效梯度信噪比高于π-StepNFT。

**关键步假设**：一条轨迹中存在少量"关键步"集合 $\mathcal{K} \subset \{1,\ldots,T\}$（$|\mathcal{K}| \ll T$），episode结果主要由关键步决定。非关键步的梯度贡献为噪声。

**分析**：

π-StepNFT的总梯度：

$$g_{\text{NFT}} = \underbrace{\sum_{i \in \mathcal{K}} \nabla_\theta \ell_{t_i}}_{\text{信号}} + \underbrace{\sum_{i \notin \mathcal{K}} \nabla_\theta \ell_{t_i}}_{\text{噪声}}$$

DECO的总梯度：

$$g_{\text{DECO}} = \underbrace{\sum_{i \in \mathcal{K}} |y_i| \cdot \nabla_\theta \ell_{t_i}}_{\text{信号（放大）}} + \underbrace{\sum_{i \notin \mathcal{K}} |y_i| \cdot \nabla_\theta \ell_{t_i}}_{\text{噪声（抑制）}}$$

关键步通常对应 $D_i$ 大（策略在难点处发生了显著变化）→ $|y_i|$ 大；非关键步对应 $D_i$ 小 → $|y_i|$ 小。

设关键步的平均权重为 $\bar{w}_\mathcal{K}$，非关键步为 $\bar{w}_\mathcal{N}$，当 $\bar{w}_\mathcal{K} > \bar{w}_\mathcal{N}$ 时：

$$\text{SNR}_{\text{DECO}} = \frac{|\mathcal{K}| \cdot \bar{w}_\mathcal{K}}{|\mathcal{N}| \cdot \bar{w}_\mathcal{N}} \cdot \text{SNR}_{\text{per-step}} > \frac{|\mathcal{K}|}{|\mathcal{N}|} \cdot \text{SNR}_{\text{per-step}} = \text{SNR}_{\text{NFT}}$$

**适用条件**：上述分析基于"关键步 ↔ 高偏差步"的对应。当条件不满足时，$w_i$ 趋向均匀，DECO退化为π-StepNFT（不会变差）。

### 4.5 与其他步级信用分配方案的理论对比

| 方案 | 步级信号来源 | 需要额外学习？ | 有偏？ | 可能比baseline差？ |
|------|------------|:---:|:---:|:---:|
| GAE（PPO） | 学习的 $V(s)$ 做TD差分 | ✅ Critic网络 | 有偏（函数逼近误差） | ✅（Critic过拟合时） |
| GRPO | 同状态多次采样比较 | ✗ | 无偏 | ✗（但方差大） |
| TGRPO | LLM生成多阶段密集奖励 | ✅ LLM推理 | 有偏（LLM判断误差） | ✅（LLM奖励不准时） |
| **DECO** | 参考模型偏差 | ✗（冻结推理） | 有偏（假设依赖） | **✗（$\eta=0$ 退化baseline）** |

---

## 五、风险评估

### 5.1 可能有效的理由

| 论据 | 置信度 |
|------|-------|
| 梯度方向与π-StepNFT一致，只调制幅度，不改变收敛方向 | **高** — 直接从Theorem 4.4推导 |
| $\eta=0$ 严格退化为baseline，不可能比π-StepNFT更差 | **高** — 严格数学等价 |
| 参考偏差在RL中被广泛验证有效（KL penalty本质相同） | **中高** — PPO/TRPO的KL约束也用ref策略 |
| Long-horizon任务的信号稀释是π-StepNFT的公认痛点 | **高** — 论文自身Sec.5局限性第1条直接指出 |
| 额外开销极小（~25%），不引入新的可学习参数 | **高** — 工程上确定 |
| π-StepNFT论文自身建议"可无缝替换为离线学习的逐步成功概率预测器" | **中** — DECO是这个方向的更简单替代 |

### 5.2 可能失败的理由

| 风险 | 严重性 | 具体分析 |
|------|--------|---------|
| "高偏差步=关键步"假设不成立 | **中** | 如果策略在所有步均匀偏离参考模型，$w_i$ 全部 $\approx 0.5$，DECO退化为π-StepNFT。**不会变差，但也不会更好。** |
| 参考模型太弱导致偏差信号无意义 | **中** | 如果SFT参考模型本身很差，所有步偏差都大，区分度消失。但π-StepNFT本身就需要SFT warm start，前提条件一致。 |
| 长轨迹中偏差随步号单调增长（累积漂移） | **中** | 如果偏离在时间维度上系统性增长（类似random walk），后期步天然获得更大权重。Batch归一化部分缓解，但不完全消除。需在实验中观察 $D_i$ vs 步号分布。 |
| $\eta$ 调参敏感 | **低** | Sigmoid归一化天然限制范围。$\eta=0$ 退化baseline，$\eta$ 过大只是放大倍数，有界性保证不爆炸。 |
| 冻结ref模型长期不更新导致信号过时 | **低中** | 随训练推进所有步偏差变大，但batch归一化确保**相对排序**仍有意义。 |

### 5.3 成功/失败的判断标准

| 结果 | 含义 | 下一步 |
|------|------|--------|
| LIBERO-Long显著优于π-StepNFT（≥ 3%），OOD不退化 | ✅ DECO有效 | 扩展到全benchmark，投稿NeurIPS |
| LIBERO-Long略优（1-2%） | ⚠️ 有帮助但不够显著 | 尝试更强偏差度量（Fisher加权）或自适应η |
| 与π-StepNFT无统计差异 | ⚠️ 假设不成立 | 分析 $D_i$ 分布——如果确实均匀，本身是有价值insight |
| 比π-StepNFT差 | ❌ 理论上不可能（$\eta$ sweep含0），说明实现有bug | 检查代码 |

---

## 六、最小可行验证实验

### 6.1 实验目标

验证一个core question：**在π-StepNFT中加入参考偏差步级权重后，LIBERO-Long上的收敛速度和最终性能是否改善，同时ManiSkill OOD不退化？**

### 6.2 Setup

| 配置项 | 选择 | 理由 |
|--------|------|------|
| 环境 | LIBERO-Long（10个长horizon子任务） | 最能暴露时间信用分配问题的benchmark |
| Policy模型 | π₀（PaliGemma-3B + ~300M action expert） | 和π-StepNFT论文一致 |
| 参考模型 | SFT checkpoint（Few-shot微调后的初始策略） | 自然的参考锚点 |
| Baselines | (1) π-StepNFT原版 (2) πRL(PPO) | 验证DECO在两者之间的定位 |
| 并行环境数 | 和π-StepNFT论文一致 | 控制变量 |
| 训练步数 | 和π-StepNFT论文一致 | 控制变量 |
| 硬件 | 8×H100 80GB | 和π-StepNFT论文一致 |

### 6.3 超参数

| 超参数 | 初始值 | 扫描范围 | 理由 |
|--------|--------|---------|------|
| 偏差调制强度 $\eta$ | 2.0 | {0, 1, 2, 3, 5} | $\eta=0$是baseline。从2开始，Long任务不够好就增大 |
| 信任域 $\beta$ | 和π-StepNFT一致 | **不扫** | 控制变量 |
| SDE噪声 $\sigma$ | 和π-StepNFT一致 | **不扫** | 控制变量 |
| 正则系数 $\lambda_{\text{TR}}$ | 和π-StepNFT一致 | **不扫** | 控制变量 |
| EMA衰减 $\alpha_m$ | 和π-StepNFT一致 | **不扫** | 控制变量 |

**关键原则**：**只扫 $\eta$**，其他全部和π-StepNFT保持一致。任何性能差异完全来自步级权重机制。

### 6.4 对比指标

| 指标 | 怎么测 | 目的 |
|------|--------|------|
| LIBERO-Long Success Rate | 每10 iter评估，每子任务50个初始状态 | 核心性能（时间信用分配效果） |
| LIBERO-Spatial/Object/Goal | 同上 | 确认短horizon任务不退化 |
| ManiSkill OOD | 同上 | 确认OOD优势不丧失（最关键的安全检查） |
| $D_i$ 分布 vs 步号 | 可视化heatmap（按成功/失败分组） | 验证"高偏差步=关键步"假设 |
| 收敛速度 | 达到85% success rate所需iteration数 | 验证信噪比改善 |
| Training curve smoothness | Success rate的10-iter滑动窗口std | 验证稳定性 |

### 6.5 代码改动清单

假设从π-StepNFT codebase出发：

1. **新增**：加载冻结ref模型（~3行）
2. **新增**：rollout时计算 $D_i$（在SDE循环内加`torch.no_grad()`块，~7行）
3. **新增**：Phase 2标签构造（batch归一化 + sigmoid + 步级标签，~5行）
4. **修改**：损失函数中 $y \to y_i$（1行）
5. **保留**：SDE rollout、镜像分支构造、步误差计算、EMA更新——全部不变

预计**总改动量 < 20行代码**。

### 6.6 时间计划

| 时间 | 任务 |
|------|------|
| Day 1 | 代码改动 + debug（改动量极小） |
| Day 2-4 | LIBERO-Long验证（$\eta = \{0, 1, 2, 3, 5\}$，5组实验并行） |
| Day 5 | $D_i$ 分布分析，理解偏差信号的时间结构 |
| Day 6 | ManiSkill OOD安全检查（用最优 $\eta$） |
| Day 7 | 分析结果，决定是否扩展到全benchmark |

---

## 七、如果验证成功 → NeurIPS 2026完整方案

### 7.1 扩展实验

| Benchmark | Tasks | 目的 |
|-----------|-------|------|
| LIBERO-Spatial/Object/Goal | 30 tasks | 验证短horizon不退化 |
| LIBERO-Long | 10 tasks | 主战场，验证长horizon改善 |
| ManiSkill IND + 3种OOD | 4352 tasks | 验证OOD优势保持 |
| ManiSkill Semantic OOD | 关键子集 | π-StepNFT最大优势点（49.1% vs PPO 25.4%），绝不能损失 |

### 7.2 消融实验

| 消融 | 目的 |
|------|------|
| $\eta \in \{0, 0.5, 1, 2, 3, 5, 10\}$ | 偏差调制强度的完整曲线 |
| Sigmoid vs Linear vs ReLU归一化 | 映射函数选择的影响 |
| $D_i$ 用全K步 vs 随机采样1步 | 偏差估计精度 vs 计算开销 |
| Batch归一化 vs 全局running statistics | 归一化策略 |
| 只在Long任务上加DECO vs 全任务加 | 验证是否对短horizon有害 |

### 7.3 分析实验

| 分析 | 方法 | 目的 |
|------|------|------|
| $D_i$ vs 步号的时间结构 | 按成功/失败分组绘制曲线 | 验证假设，理解偏差的时间行为 |
| 高权重步的语义可视化 | 渲染 $|y_i|$ top-10步的机器人画面 | 定性验证"关键步"是否确实关键 |
| DECO权重 vs PPO的GAE优势 | 在同一batch上算Spearman相关 | 量化DECO权重作为信用分配近似的质量 |
| 训练过程中 $D_i$ 分布演化 | 每隔若干iter绘制直方图 | 理解信号是否随训练退化 |

### 7.4 理论贡献

1. **梯度方向保持定理**（命题1）：证明DECO保持π-StepNFT的oracle对齐性
2. **有界性保证**（命题2）：$|y_i| \in [1, 1+\eta]$，无爆炸无消失
3. **退化性保证**（命题3）：$\eta=0$ 严格回退基线
4. **信噪比改善**（命题4）：在关键步假设下的定性分析
5. **（可选）** 参考偏差与真实优势函数 $|A(s_i, a_i)|$ 正相关的条件分析

### 7.5 论文结构

**Title:** *DECO: Deviation-Enhanced Contrastive Optimization for Critic-Free RL in Flow-Based VLAs*

| Section | Pages | 内容 |
|---------|-------|------|
| 1. Introduction | 1.5 | π-StepNFT的OOD优势 → Long-horizon瓶颈 → 参考偏差动机 |
| 2. Background | 1 | Flow matching, π-StepNFT回顾, 信用分配问题定义 |
| 3. DECO | 1.5 | 方法 + Algorithm + 15行代码实现 |
| 4. Theoretical Analysis | 1 | 梯度保持 + 有界性 + 退化性 + 信噪比 |
| 5. Experiments | 3 | 主实验 + 消融 + $D_i$ 分布可视化分析 |
| 6. Related Work | 0.5 | Critic-free RL + 信用分配全景 |
| 7. Conclusion | 0.5 | |

### 7.6 与现有方法的定位

| 方法 | Critic | 似然 | 在线 | Flow VLA | 步级信用 | OOD泛化 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| πRL (PPO) | ✅ | 近似 | ✓ | ✓ | ✓ (GAE) | ⚠️ −11.1% |
| TGRPO | ✗ | ✗ | ✓ | ✗ (AR only) | ✓ (LLM奖励) | 未测 |
| FPO++ | ✗ | 近似 | ✓ | ✓ | ✗ | 未测 |
| π-StepNFT | ✗ | ✗ | ✓ | ✓ | ✗ | ✅ |
| **DECO** | **✗** | **✗** | **✓** | **✓** | **✓** | **✅** |

DECO的定位：**π-StepNFT的严格超集。保留其所有优势（无Critic、无似然、OOD泛化），通过一个等式修改增加步级时间信用分配。**

---

## 八、可选扩展（如果基础版有效）

**扩展A：自适应 $\eta$ 调度**

训练初期策略接近参考模型，$D_i$ 普遍小、区分度不够。随训练推进逐步增大 $\eta$：

$$\eta_k = \eta_{\min} + (\eta_{\max} - \eta_{\min})\cdot\min(1, k/K_{\text{warmup}})$$

**扩展B：Chunk内逐步偏差**

当前 $D_i$ 是环境步级别的。如果action chunk = 16步，可以进一步分解为chunk内逐步偏差，对chunk内每个动作维度分配不同权重。但这需要改动π-StepNFT的chunk处理逻辑，复杂度增加。

**扩展C：多参考点集成**

维护历史checkpoint队列，取多参考点的最大偏差：

$$D_i^{\text{multi}} = \max_{k \in \text{queue}} \left\|v_{\theta_{\text{ref}}^{(k)}} - v_{\theta_{\text{old}}}\right\|^2$$

确保关键步不会因为某个参考点而被漏掉。

---

## 九、总结

**一句话**：π-StepNFT在长horizon任务上被PPO拉开（86.7% vs 90.2%），根因是episode级二值标签提供零时间信用分配。DECO用**参考模型偏差**作为步级权重（$y \to y_i$），保持π-StepNFT的所有优势（无Critic、无似然、OOD +11.1%），同时提供步级信用分配。

**最小可行验证**：在LIBERO-Long上，改约15行代码，1周内可知结论。

**最大风险**："高偏差步=关键步"假设不成立，此时DECO退化为π-StepNFT，不会变差。

**最好情况**：LIBERO-Long显著提升 + OOD保持 → NeurIPS 2026投稿。

**最差情况**：与π-StepNFT无差异 → 说明flow VLA策略演化在时间维度上均匀分布，偏差不集中于少数步 → 有价值的insight，意味着信用分配问题可能需要从奖励端而非权重端解决。
