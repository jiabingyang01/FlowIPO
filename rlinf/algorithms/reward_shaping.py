"""VP-PPO: VLM-Potential Reward Shaping for Flow-based VLA RL.

Uses frozen VLM embedding cosine similarity with success-state targets
as a PBRS (Potential-Based Reward Shaping) potential function.
PBRS theorem (Ng et al., 1999) guarantees optimal policy invariance.
"""

from collections import defaultdict

import torch
import torch.nn.functional as F


class SuccessFeatureBuffer:
    """Per-task EMA buffer of success-trajectory final-frame VLM embeddings.

    Maintains a running EMA target for each task (identified by hash).
    Used by VP-PPO to construct the PBRS potential function target.
    """

    def __init__(self, ema_rate: float = 0.99, min_count: int = 5):
        self.ema_rate = ema_rate
        self.min_count = min_count
        self.ema_target: dict[int, torch.Tensor] = {}
        self.count: dict[int, int] = defaultdict(int)

    def update(self, task_hash: int, success_final_embs: torch.Tensor):
        """Update EMA target with successful episode final-frame embeddings.

        Args:
            task_hash: Task identifier (language token hash or 0 for single-task).
            success_final_embs: [n_success, hidden_dim] tensor of final-frame VLM embeddings.
        """
        if success_final_embs.numel() == 0:
            return
        batch_mean = success_final_embs.mean(dim=0).detach()
        if task_hash not in self.ema_target:
            self.ema_target[task_hash] = batch_mean.clone()
        else:
            self.ema_target[task_hash] = (
                self.ema_rate * self.ema_target[task_hash]
                + (1.0 - self.ema_rate) * batch_mean
            )
        self.count[task_hash] += success_final_embs.shape[0]

    def get_target(self, task_hash: int) -> torch.Tensor | None:
        """Return EMA target embedding, or None if cold-starting."""
        return self.ema_target.get(task_hash, None)

    def has_enough(self, task_hash: int) -> bool:
        """Whether enough successful episodes have been collected."""
        return self.count.get(task_hash, 0) >= self.min_count

    def state_dict(self) -> dict:
        """For checkpointing."""
        return {
            "ema_target": {k: v.cpu() for k, v in self.ema_target.items()},
            "count": dict(self.count),
        }

    def load_state_dict(self, state: dict):
        """Restore from checkpoint."""
        self.ema_target = {k: v for k, v in state.get("ema_target", {}).items()}
        self.count = defaultdict(int, state.get("count", {}))


def compute_vlm_potential(
    vlm_embs: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute potential function Φ(s_t) = cosine_similarity(emb_t, target).

    Args:
        vlm_embs: [n_steps, batch, hidden_dim] VLM embeddings per step.
        target: [hidden_dim] EMA success target embedding.

    Returns:
        phi: [n_steps, batch] potential values in [-1, 1].
    """
    # target: [hidden_dim] -> [1, 1, hidden_dim]
    target_expanded = target.unsqueeze(0).unsqueeze(0).expand_as(vlm_embs)
    phi = F.cosine_similarity(vlm_embs, target_expanded, dim=-1)
    return phi  # [n_steps, batch]


def compute_pbrs_reward(
    env_rewards: torch.Tensor,
    phi: torch.Tensor,
    gamma: float = 0.99,
    alpha: float = 0.2,
) -> torch.Tensor:
    """Apply PBRS: r'_t = r_t + alpha * (gamma * Phi(s_{t+1}) - Phi(s_t)).

    Args:
        env_rewards: [n_steps, batch] original environment rewards.
        phi: [n_steps, batch] potential function values.
        gamma: Discount factor (same as PPO gamma).
        alpha: PBRS scaling coefficient.

    Returns:
        shaped_rewards: [n_steps, batch] shaped rewards.
    """
    n_steps = phi.shape[0]
    # Phi(s_{t+1}): shift phi by 1 step; for last step use phi[-1] (terminal)
    phi_next = torch.cat([phi[1:], phi[-1:]], dim=0)  # [n_steps, batch]
    pbrs = alpha * (gamma * phi_next - phi)
    return env_rewards + pbrs
