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
FlowIPO credit assignment module.

Computes per-step interpolation weights w_i based on:
  1. Policy divergence: δ_i = ||a_i - a_i^ref||_2
  2. Reward-directed credit: A_i = (2R - 1) * normalize(δ_i)
  3. Interpolation weight: w_i = sigmoid(α * A_i) ∈ (0, 1)

w_i → 1: reinforce current action (success + large divergence)
w_i → 0: fall back to reference policy (failure + large divergence)
"""

import torch


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
