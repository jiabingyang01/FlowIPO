# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Credit assignment module for FlowIPO, FlowSAR, and FPI.

FlowIPO:
  Computes per-step interpolation weights w_i based on:
    1. Policy divergence: δ_i = ||a_i - a_i^ref||_2
    2. Reward-directed credit: A_i = (2R - 1) * normalize(δ_i)
    3. Interpolation weight: w_i = sigmoid(α * A_i) ∈ (0, 1)

FlowSAR:
  Computes per-step credit assignment weights based on:
    1. Reconstruction error e_i (policy confidence proxy)
    2. Success: w_i = softmax(e_i / T) — uncertain but correct steps get high weight
    3. Failure: w_i = softmax(-e_i / T) — confident but wrong steps get high weight
"""

import math

import torch
import torch.nn.functional as F


def compute_flow_ipo_weights(
    actions: torch.Tensor,
    ref_actions: torch.Tensor,
    rewards: torch.Tensor,
    alpha: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute FlowIPO interpolation weights for each environment step.

    Args:
        actions: Rollout actions, shape [n_steps, batch, action_dim] or [batch, action_dim].
        ref_actions: Reference policy actions from same initial noise,
                     same shape as actions.
        rewards: Episode-level reward, shape [batch] (binary or shaped).
        alpha: Temperature controlling weight sharpness.
        eps: Small constant for numerical stability in normalization.

    Returns:
        weights: Per-step interpolation weights, shape matching actions[..., :1]
                 (last dim squeezed to 1 for broadcasting).
    """
    # δ_i = ||a_i - a_i^ref||_2, per-step policy divergence
    # actions 和 ref_actions 的 shape 可能是 [n_steps, batch, chunk, action_dim]
    # 或 [batch, chunk, action_dim]，对最后两维求 L2 norm
    delta = (actions - ref_actions).float()
    # Flatten chunk and action dims for norm computation
    orig_shape = delta.shape
    if delta.dim() >= 3:
        # Compute L2 norm over chunk*action_dim dimensions
        delta_flat = delta.reshape(*orig_shape[:-2], -1)  # [..., chunk * action_dim]
        delta_norm = torch.norm(delta_flat, dim=-1)  # [...] per-step scalar
    else:
        delta_norm = torch.norm(delta, dim=-1)

    # Episode-level normalization: A_i = (2R - 1) * (δ_i - mean) / (std + ε)
    # rewards shape: [batch] -> broadcast to match delta_norm
    reward_sign = (2.0 * rewards.float() - 1.0)  # ∈ {-1, +1} for binary rewards

    # Normalize δ within each episode (across steps)
    if delta_norm.dim() >= 2:
        # [n_steps, batch] -> normalize across steps (dim=0)
        mean_delta = delta_norm.mean(dim=0, keepdim=True)
        std_delta = delta_norm.std(dim=0, keepdim=True)
        normalized_delta = (delta_norm - mean_delta) / (std_delta + eps)
        # reward_sign: [batch] -> [1, batch]
        credit = reward_sign.unsqueeze(0) * normalized_delta
    else:
        # Single step case: [batch]
        mean_delta = delta_norm.mean()
        std_delta = delta_norm.std()
        normalized_delta = (delta_norm - mean_delta) / (std_delta + eps)
        credit = reward_sign * normalized_delta

    # w_i = sigmoid(α * A_i) ∈ (0, 1)
    weights = torch.sigmoid(alpha * credit)

    return weights


def compute_flow_sar_weights(
    recon_errors: torch.Tensor,
    rewards: torch.Tensor,
    temperature: float = 0.5,
    w_min: float = 0.0,
    w_max: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute FlowSAR step-level credit assignment weights based on
    reconstruction error and episode reward.

    Success trajectories (R=1): uncertain-but-correct steps get high weight
        w_i = softmax(e_i / T)
    Failure trajectories (R=0): confident-but-wrong steps get high weight
        w_i = softmax(-e_i / T)

    Args:
        recon_errors: Per-step reconstruction error, shape [n_steps, batch].
        rewards: Episode-level reward, shape [batch] (binary 0/1).
        temperature: Softmax temperature controlling weight sharpness.
        w_min: Minimum weight for clipping (0 = no clipping).
        w_max: Maximum weight for clipping (1 = no clipping).
        eps: Small constant for numerical stability.

    Returns:
        weights: Per-step credit assignment weights, shape [n_steps, batch].
        labels: Per-sample contrastive labels y_i = 2R - 1, shape [batch].
    """
    n_steps, batch_size = recon_errors.shape
    rewards_float = rewards.float()

    # Contrastive labels: y_i = 2R - 1 ∈ {-1, +1}
    labels = 2.0 * rewards_float - 1.0  # [batch]

    # Success mask and failure mask
    success_mask = (rewards_float > 0.5)  # [batch]
    failure_mask = ~success_mask

    # Compute softmax weights per episode (across steps)
    # recon_errors: [n_steps, batch] -> transpose to [batch, n_steps] for softmax
    errors_t = recon_errors.float().transpose(0, 1)  # [batch, n_steps]

    weights = torch.zeros_like(errors_t)  # [batch, n_steps]

    # Success: softmax(e_i / T) — high error = high weight
    if success_mask.any():
        success_logits = errors_t[success_mask] / (temperature + eps)
        weights[success_mask] = F.softmax(success_logits, dim=-1)

    # Failure: softmax(-e_i / T) — low error (high confidence) = high weight
    if failure_mask.any():
        failure_logits = -errors_t[failure_mask] / (temperature + eps)
        weights[failure_mask] = F.softmax(failure_logits, dim=-1)

    # Optional weight clipping and re-normalization
    if w_min > 0.0 or w_max < 1.0:
        weights = weights.clamp(min=w_min, max=w_max)
        # Re-normalize so weights sum to 1 per episode
        weight_sums = weights.sum(dim=-1, keepdim=True).clamp(min=eps)
        weights = weights / weight_sums

    # Scale weights so that mean weight = 1 (instead of 1/n_steps)
    # This makes the loss magnitude independent of episode length
    weights = weights * n_steps

    # Transpose back to [n_steps, batch]
    weights = weights.transpose(0, 1)

    return weights, labels


def compute_flow_awm_advantages(
    advantages: torch.Tensor,
    clip_range: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute AWM-VLA normalized and clipped advantages (linear mode, can be negative).

    Ã = clip( (A - mean(A)) / (std(A) + ε),  -c,  c )

    - Ã > 0: pull velocity toward good action's target
    - Ã < 0: push velocity away from bad action's target

    Args:
        advantages: Raw per-step advantages, shape [n_steps, batch].
        clip_range: Symmetric clipping threshold c. Ã ∈ [-c, c].
        eps: Small constant for numerical stability.

    Returns:
        clipped_advantages: Normalized and clipped advantages, shape [n_steps, batch].
    """
    adv = advantages.float()

    # Global normalization across all steps and batch
    adv_mean = adv.mean()
    adv_std = adv.std().clamp(min=eps)
    adv_normalized = (adv - adv_mean) / adv_std

    # Symmetric clipping to prevent extreme push/pull
    adv_clipped = adv_normalized.clamp(-clip_range, clip_range)

    return adv_clipped


def compute_flow_awm_exp_weights(
    advantages: torch.Tensor,
    beta: float = 1.0,
    max_weight: float = 6.0,
    normalize_adv: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute AWM-VLA exponential advantage weights (AWR-style, always positive).

    w = clamp( exp(A / β),  max=w_max )

    All weights are positive, so all samples pull velocity TOWARD their target.
    Good actions get large weights, bad actions get small (but > 0) weights.
    No gradient cancellation.

    Args:
        advantages: Raw per-step advantages, shape [n_steps, batch].
        beta: Temperature controlling weight sharpness (smaller = more greedy).
        max_weight: Upper clamp to prevent single samples from dominating.
        normalize_adv: Whether to standardize advantages before exponentiation.
        eps: Small constant for numerical stability.

    Returns:
        weights: Positive advantage weights, shape [n_steps, batch].
    """
    adv = advantages.float()

    if normalize_adv:
        adv_mean = adv.mean()
        adv_std = adv.std().clamp(min=eps)
        adv = (adv - adv_mean) / adv_std

    weights = torch.clamp(torch.exp(adv / beta), max=max_weight)

    return weights


def compute_gfn_log_pf(
    chain: torch.Tensor,
    velocities: torch.Tensor,
    timesteps: torch.Tensor,
    sigma_f: float,
    action_dim: int,
) -> torch.Tensor:
    """
    Compute log forward transition probabilities for GFN-Flow.

    Forward policy: P_F(x_{k+1} | x_k) = N(μ_k, σ_f²I)
    where μ_k = x_k - v_θ(x_k, t_k) * δ_k  (Euler step, code convention t=1→noise)
    δ_k = t_k - t_{k+1} > 0

    Args:
        chain: Denoising chain states, shape [batch, K+1, horizon, dim].
        velocities: Predicted velocities at each step, shape [batch, K, horizon, dim].
        timesteps: Timestep values, shape [K+1].
        sigma_f: Forward policy noise standard deviation.
        action_dim: Total action dimension (horizon * dim) for log-prob computation.

    Returns:
        log_pf: Log forward transition probs, shape [batch, K].
    """
    K = velocities.shape[1]
    bsz = chain.shape[0]
    log_pf_list = []

    for k in range(K):
        delta_k = timesteps[k] - timesteps[k + 1]  # > 0 (t decreasing)
        # Euler step: μ_k = x_k - v_θ * δ_k
        mu_k = chain[:, k] - velocities[:, k] * delta_k  # [bsz, horizon, dim]
        x_next = chain[:, k + 1]  # [bsz, horizon, dim]

        # log N(x_{k+1}; μ_k, σ_f²I) summed over all action dimensions
        diff = (x_next - mu_k) / sigma_f  # [bsz, horizon, dim]
        log_p = -0.5 * diff.pow(2).reshape(bsz, -1).sum(dim=-1)  # [bsz]
        log_p = log_p - action_dim * (0.5 * math.log(2 * math.pi) + math.log(sigma_f))
        log_pf_list.append(log_p)

    return torch.stack(log_pf_list, dim=1)  # [bsz, K]


def compute_gfn_log_pb(
    chain: torch.Tensor,
    timesteps: torch.Tensor,
    sigma_b: float,
    action_dim: int,
) -> torch.Tensor:
    """
    Compute log backward transition probabilities for GFN-Flow.

    Backward policy (FIXED, no learnable params):
    P_B(x_k | x_{k+1}) = N(μ_B, σ_B²I)
    where μ_B = ((1-t_k) / (1-t_{k+1})) * x_{k+1}  (rectified flow backward scaling)

    In code convention (t=1→noise, t=0→clean):
    - (1-t) represents the "clean fraction" of the interpolation
    - ratio = (1-t_k) / (1-t_{k+1}) < 1 when t_k > t_{k+1}
    - Special: t_k=1 → ratio=0 → μ_B=0 (backward to pure noise)

    Args:
        chain: Denoising chain states, shape [batch, K+1, horizon, dim].
        timesteps: Timestep values, shape [K+1].
        sigma_b: Backward policy noise standard deviation.
        action_dim: Total action dimension (horizon * dim) for log-prob computation.

    Returns:
        log_pb: Log backward transition probs, shape [batch, K].
    """
    K = chain.shape[1] - 1
    bsz = chain.shape[0]
    log_pb_list = []

    for k in range(K):
        t_k = timesteps[k]
        t_kp1 = timesteps[k + 1]

        # Backward ratio: (1 - t_k) / (1 - t_{k+1})
        # t_k > t_{k+1} (decreasing), so (1-t_k) < (1-t_{k+1}), ratio < 1
        clean_frac_k = 1.0 - t_k
        clean_frac_kp1 = 1.0 - t_kp1
        ratio = clean_frac_k / max(clean_frac_kp1, 1e-6)

        # μ_B = ratio * x_{k+1}
        mu_b = ratio * chain[:, k + 1]  # [bsz, horizon, dim]
        x_k = chain[:, k]  # [bsz, horizon, dim]

        # log N(x_k; μ_B, σ_b²I) summed over all action dimensions
        diff = (x_k - mu_b) / sigma_b
        log_p = -0.5 * diff.pow(2).reshape(bsz, -1).sum(dim=-1)  # [bsz]
        log_p = log_p - action_dim * (0.5 * math.log(2 * math.pi) + math.log(sigma_b))
        log_pb_list.append(log_p)

    return torch.stack(log_pb_list, dim=1)  # [bsz, K]


def compute_terminal_binary_advantages(
    rewards: torch.Tensor,
    n_steps: int,
    adv_clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Map episode success/failure to ±adv_clip_max for all env steps.

    Success (reward > 0.5): advantage = +adv_clip_max for all steps
    Failure (reward <= 0.5): advantage = -adv_clip_max for all steps

    Args:
        rewards: Episode-level reward, shape [batch]. Binary (0 or 1).
        n_steps: Number of environment steps per episode.
        adv_clip_max: Advantage magnitude for ±1 labels.

    Returns:
        advantages: Per-step advantages, shape [n_steps, batch].
    """
    success = (rewards.float() > 0.5).float()  # [batch]
    labels = success * 2 - 1  # +1 success, -1 failure
    advantages = labels * adv_clip_max  # [batch]
    # Broadcast to all steps
    advantages = advantages.unsqueeze(0).expand(n_steps, -1)  # [n_steps, batch]
    return advantages


def compute_frozen_embedding_advantages(
    embeddings: torch.Tensor,
    episode_rewards: torch.Tensor,
    gamma: float = 0.99,
    ridge_lambda: float = 1.0,
    adv_clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Frozen Embedding Advantage (FEA): Ridge regression on frozen VLM embeddings
    for per-step temporal credit assignment.

    The DIRECTION (sign) always comes from terminal_binary (success=+, failure=-).
    FEA only modulates the MAGNITUDE per step via importance weights derived
    from a ridge-regression value function on frozen VLM embeddings.

    This avoids the failure mode where normalized TD advantages produce random
    signs that conflict with the DPO softplus loss (which is designed for
    binary labels and amplifies wrong-sign signals through large delta_E).

    Pipeline:
        1. Ridge regression: V(e) = w^T e + b, fitted to MC returns G_t = γ^(T-1-t) R
        2. Per-step importance: |ΔV_t| = |γ V(e_{t+1}) - V(e_t)|  (value change)
        3. Normalize importance to mean 1, clip to [w_min, w_max]
        4. Final advantage = terminal_binary_sign × importance × adv_clip_max

    Args:
        embeddings: Frozen VLM embeddings, shape [n_steps, batch, hidden_dim].
        episode_rewards: Episode-level reward, shape [batch]. Binary (0 or 1).
        gamma: Temporal discount factor.
        ridge_lambda: Ridge regression regularization.
        adv_clip_max: Advantage clipping range.

    Returns:
        advantages: Per-step advantages, shape [n_steps, batch].
                    Sign matches terminal_binary; magnitude modulated by FEA.
    """
    T, B, d = embeddings.shape
    device = embeddings.device
    dtype = embeddings.dtype

    # ---- terminal binary direction (never overridden) ----
    success = (episode_rewards.float() > 0.5).float()
    terminal_sign = (success * 2 - 1).unsqueeze(0).expand(T, -1)  # [T, B], ±1

    # ---- ridge regression value function ----
    # MC return targets: G_t = γ^(T-1-t) × R
    gammas = gamma ** torch.arange(T - 1, -1, -1, device=device, dtype=dtype)  # [T]
    targets = gammas.unsqueeze(1) * episode_rewards.unsqueeze(0).float()  # [T, B]

    E = embeddings.reshape(-1, d).float()  # [T*B, d]
    E_bias = torch.cat([E, torch.ones(E.shape[0], 1, device=device, dtype=E.dtype)], dim=1)
    G = targets.reshape(-1)  # [T*B]

    EtE = E_bias.T @ E_bias + ridge_lambda * torch.eye(d + 1, device=device, dtype=E.dtype)
    w = torch.linalg.solve(EtE, E_bias.T @ G)  # [d+1]

    V = (E_bias @ w).reshape(T, B)  # [T, B]

    # ---- per-step importance from |TD error| ----
    td = torch.zeros_like(V)
    td[:-1] = gamma * V[1:] - V[:-1]
    td[-1] = episode_rewards.float() - V[-1]

    importance = td.abs()  # [T, B]
    imp_mean = importance.mean().clamp(min=1e-8)
    importance = importance / imp_mean  # normalize to mean ≈ 1
    w_min, w_max = 0.2, 2.0
    importance = importance.clamp(w_min, w_max)  # prevent extreme weights

    # ---- combine: terminal direction × FEA importance ----
    advantages = terminal_sign * importance * adv_clip_max

    return advantages


def compute_embedding_change_advantages(
    embeddings: torch.Tensor,
    episode_rewards: torch.Tensor,
    n_steps: int,
    adv_clip_max: float = 1.0,
    w_min: float = 0.2,
    w_max: float = 2.0,
) -> torch.Tensor:
    """
    VLM Embedding Change Rate credit assignment for Hinge-NFT (方案B).

    Uses the frozen VLM embedding's frame-to-frame change rate as step
    importance weight. Core insight: scene-changing frames (grasp, place)
    correspond to critical decision points and have large ||e_{t+1} - e_t||.

    The SIGN always comes from terminal binary (correct direction guaranteed).
    The MAGNITUDE is modulated by embedding change rate (critical steps get
    higher weight, routine steps get lower weight).

    This is safe with hinge loss because:
      - Sign from terminal binary → direction always correct
      - Continuous |y| values work with hinge (symmetric gradient)
      - Zero extra cost: embeddings already collected during rollout

    Args:
        embeddings: Frozen VLM embeddings, shape [n_steps, batch, hidden_dim].
        episode_rewards: Episode-level reward, shape [batch]. Binary (0 or 1).
        n_steps: Number of environment steps per episode.
        adv_clip_max: Advantage clipping range.
        w_min: Min importance weight (ensures minimum signal, default 0.2).
        w_max: Max importance weight (caps extreme values, default 2.0).

    Returns:
        advantages: Per-step advantages, shape [n_steps, batch].
                    Sign from terminal binary; magnitude from embedding change.
    """
    batch_size = episode_rewards.shape[0]

    # ---- terminal binary direction ----
    success = (episode_rewards.float() > 0.5).float()  # [batch]
    terminal_sign = success * 2 - 1  # +1 success, -1 failure
    terminal_sign = terminal_sign.unsqueeze(0).expand(n_steps, -1)  # [n_steps, batch]

    # ---- per-step importance from embedding change rate ----
    # embeddings: [n_steps, batch, hidden_dim]
    # Compute frame-to-frame L2 distance: ||e_{t+1} - e_t||
    emb_diff = embeddings[1:] - embeddings[:-1]  # [n_steps-1, batch, hidden_dim]
    change_rate = emb_diff.norm(dim=-1)  # [n_steps-1, batch]

    # Pad last step (no next frame) with the mean change rate
    pad = change_rate.mean(dim=0, keepdim=True)  # [1, batch]
    change_rate = torch.cat([change_rate, pad], dim=0)  # [n_steps, batch]

    # Normalize to mean=1 per episode, then clamp
    w_mean = change_rate.mean(dim=0, keepdim=True).clamp(min=1e-8)  # [1, batch]
    importance = change_rate / w_mean  # mean ≈ 1 per episode
    importance = importance.clamp(w_min, w_max)

    # ---- combine: terminal direction × importance × scale ----
    advantages = terminal_sign * importance * adv_clip_max

    return advantages


def compute_deco_advantages(
    episode_rewards: torch.Tensor,
    ref_deviations: torch.Tensor,
    n_steps: int,
    adv_clip_max: float = 1.0,
    eta: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """
    DECO: Deviation-Enhanced Contrastive Optimization credit assignment.

    Uses the reference model deviation D_i as a step-level importance weight.
    Core insight: steps where the current policy deviates most from the frozen
    reference (SFT checkpoint) are likely the key decision points.

    The SIGN always comes from terminal binary (direction always correct).
    The MAGNITUDE is modulated by sigmoid-normalized deviation weight.

    Formula:
        w_i = sigmoid((D_i - D_mean) / D_std)       ∈ (0, 1)
        y_i = (2r - 1) * (1 + η * w_i) * adv_clip_max

    When η=0, strictly degenerates to π-StepNFT terminal binary.

    Args:
        episode_rewards: Episode-level reward, shape [batch]. Binary (0 or 1).
        ref_deviations: Per-step reference deviation D_i, shape [n_steps, batch].
            D_i = ||v_ref - v_old||^2 at sampled denoising step.
        n_steps: Number of environment steps per episode.
        adv_clip_max: Advantage clipping range.
        eta: Deviation modulation strength (only new hyperparameter).
            η=0 → degenerates to terminal binary.
            η=2 → max weight ratio ≈ 3x.

    Returns:
        advantages: Per-step advantages, shape [n_steps, batch].
        metrics: Dict with DECO-specific diagnostic metrics.
    """
    batch_size = episode_rewards.shape[0]
    device = episode_rewards.device

    # ---- terminal binary direction (never overridden) ----
    success = (episode_rewards.float() > 0.5).float()  # [batch]
    terminal_sign = success * 2 - 1  # +1 success, -1 failure
    terminal_sign = terminal_sign.unsqueeze(0).expand(n_steps, -1)  # [n_steps, batch]

    # ---- batch normalization of deviations ----
    D_mean = ref_deviations.mean()
    D_std = ref_deviations.std().clamp(min=1e-8)
    w = torch.sigmoid((ref_deviations - D_mean) / D_std)  # [n_steps, batch], ∈ (0, 1)

    # ---- step-level label: y_i = (2r-1) * (1 + η * w_i) * adv_clip_max ----
    advantages = terminal_sign * (1.0 + eta * w) * adv_clip_max  # [n_steps, batch]

    # ---- diagnostic metrics ----
    metrics = {
        "deco/D_mean": D_mean.item(),
        "deco/D_std": D_std.item(),
        "deco/w_mean": w.mean().item(),
        "deco/w_std": w.std().item(),
        "deco/w_min": w.min().item(),
        "deco/w_max": w.max().item(),
        "deco/y_abs_mean": advantages.abs().mean().item(),
        "deco/y_abs_max": advantages.abs().max().item(),
        "deco/eta": eta,
    }

    return advantages, metrics


def compute_flow_fpi_weights(
    advantages: torch.Tensor,
    lambda_: float = 1.0,
    w_min: float = 0.0,
    w_max: float = 10.0,
    normalize_adv: bool = True,
) -> torch.Tensor:
    """
    Compute FPI advantage weights via self-normalized exponential tilting.

    w_i = clip( exp(A_i / λ) / mean_j(exp(A_j / λ)),  w_min,  w_max )

    With normalize_adv=True, advantages are first standardized to zero-mean
    unit-variance.  This prevents weight explosion as value network improves
    and advantage variance grows.

    Args:
        advantages: Per-step advantages, shape [n_steps, batch].
        lambda_: Temperature controlling exploitation (smaller = more greedy).
        w_min: Minimum weight after clipping.
        w_max: Maximum weight after clipping.
        normalize_adv: If True, standardize advantages before exponentiation.

    Returns:
        weights: Self-normalized advantage weights, shape [n_steps, batch].
    """
    adv = advantages.float()

    # Standardize advantages to prevent weight explosion as value improves
    if normalize_adv:
        adv_mean = adv.mean()
        adv_std = adv.std().clamp(min=1e-6)
        adv = (adv - adv_mean) / adv_std

    # Transpose to [batch, n_steps] for per-episode normalization
    adv_t = adv.transpose(0, 1)  # [batch, n_steps]

    # Log-space computation for numerical stability
    log_w = adv_t / (lambda_ + 1e-8)
    log_w = log_w - log_w.max(dim=-1, keepdim=True).values  # shift for stability

    w = torch.exp(log_w)

    # Self-normalization: mean weight = 1 per episode
    w = w / w.mean(dim=-1, keepdim=True).clamp(min=1e-8)

    # Clip extreme weights
    if w_min > 0.0 or w_max < float("inf"):
        w = w.clamp(min=w_min, max=w_max)

    # Transpose back to [n_steps, batch]
    weights = w.transpose(0, 1)

    return weights
