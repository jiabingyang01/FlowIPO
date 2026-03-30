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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep conditioning."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep values, shape [batch] or [batch, 1].
        Returns:
            Sinusoidal embedding, shape [batch, embed_dim].
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [batch, 1]
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t * freqs.unsqueeze(0)  # [batch, half_dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [batch, embed_dim]
        return emb


class StateFlowNet(nn.Module):
    """
    State flow estimator F_psi for GFN-Flow.

    Estimates log F_psi(x_k, t_k, s) at each intermediate denoising state.
    The state flow represents the "potential" of a denoising intermediate state
    for reaching high-reward terminal actions.

    Input:
        - suffix_out_pooled: Pooled action-expert features [batch, input_dim].
          Contains both observation info (via VLM KV cache) and action state info.
        - timestep: Denoising timestep [batch], code convention (t=1→noise, t=0→clean).

    Output:
        - log_flow: [batch] scalar (apply softplus externally for positive flow).

    Architecture: MLP with sinusoidal timestep embedding.
    """

    def __init__(
        self,
        input_dim: int,
        time_embed_dim: int = 64,
        hidden_sizes: tuple = (512, 256, 128),
        activation: str = "gelu",
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        total_input = input_dim + time_embed_dim
        layers = []
        in_dim = total_input

        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h

        layers.append(nn.Linear(in_dim, 1, bias=True))
        self.mlp = nn.Sequential(*layers)

        self._init_weights(activation.lower())

    def _init_weights(self, nonlinearity="gelu"):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    # Small init for output layer
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self, suffix_out_pooled: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            suffix_out_pooled: [batch, input_dim] from action expert (mean-pooled).
            timestep: [batch] denoising timestep (code convention).

        Returns:
            log_flow: [batch] raw log-flow value. Apply softplus for positive flow.
        """
        t_emb = self.time_embed(timestep)  # [batch, time_embed_dim]
        x = torch.cat([suffix_out_pooled, t_emb], dim=-1)
        return self.mlp(x).squeeze(-1)  # [batch]
