import torch
from typing import Optional
from src.energies.base_energy_function import BaseEnergyFunction

_EPSILON = 1e-6


class Clipper:
    def __init__(
        self,
        should_clip_scores: bool,
        should_clip_log_rewards: bool,
        max_score_norm: Optional[float] = None,
        min_log_reward: Optional[float] = None,
        energy_function: Optional[BaseEnergyFunction] = None,
    ):
        self._should_clip_scores = should_clip_scores
        self._should_clip_log_rewards = should_clip_log_rewards
        self.max_score_norm = max_score_norm
        self.min_log_reward = min_log_reward
        self.energy_function = energy_function

    @property
    def should_clip_scores(self) -> bool:
        return self._should_clip_scores

    @property
    def should_clip_log_rewards(self) -> bool:
        return self._should_clip_log_rewards
    
    @property
    def is_molecule(self) -> bool:
        if self.energy_function is not None:
            return self.energy_function.is_molecule
        return False

    def clip_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if self.is_molecule:
            scores = scores.reshape(
                -1,
                self.energy_function.n_particles,
                self.energy_function.n_spatial_dim,
            )

        score_norms = torch.linalg.vector_norm(scores, dim=-1).detach()

        clip_coefficient = torch.clamp(
            self.max_score_norm / (score_norms + _EPSILON), max=1
        )

        clipped_scores = scores * clip_coefficient.unsqueeze(-1)

        if self.is_molecule:
            clipped_scores = clipped_scores.reshape(-1,
                                                    self.energy_function.dimensionality
            )
        return clipped_scores

    def clip_log_rewards(self, log_rewards: torch.Tensor) -> torch.Tensor:
        return log_rewards.clamp(min=self.min_log_reward)
    
    def wrap_grad_fxn(self, grad_fxn):
        def _run(*args, **kwargs):
            scores = grad_fxn(*args, **kwargs)
            if self.should_clip_scores:
                scores = self.clip_scores(scores)

            return scores

        return _run
