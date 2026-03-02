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
Credit assignment module for FlowIPO and FlowSAR.

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
