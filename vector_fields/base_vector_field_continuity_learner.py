from abc import ABC, abstractmethod
from torchtyping import TensorType
import torch

class BaseVectorFieldContinuityLearner(ABC):
    def __init__(self, vector_field: VectorField, error_norm_ord: float = 2.0):
        self.vector_field = vector_field
        self.error_norm_ord = error_norm_ord

    def loss(self, x: TensorType['batch_size', 'traj_len', 'data_dim']) -> TensorType['batch_size']:
        '''
        Given some values x and times t, computes the continuity equation error

            || dt log pt(x) + <grad log pt(x), vt(x)> + div(vt(x)) ||^2

        Assumes that x and t have the same length in first dimension
        '''
        t = torch.linspace(0, 1, x.shape[1], device=x.device)
        x = x.flatten(0, 1)

        vt = self.vector_field(x, t)

        dt_log_pt = self.get_dt_log_pt(x, t)
        score_vt_inner_prod = self.get_score_vt_inner_product(x, t, vt)
        vt_divergence = self.get_vt_divergence(x, t, vt)
        other_terms = self.get_extra_terms(x, t, vt)

        pre_error = dt_log_pt + score_vt_inner_prod + vt_divergence + other_terms
        return torch.linalg.norm(pre_error, ord=self.error_norm_ord, dim=-1)

    @abstracthmethod
    def get_dt_log_pt(
        self,
        x: TensorType['batch_size', 'data_dim'],
        t: TensorType['batch_size']
    ) -> TensorType['batch_size']:
        '''
        Given some values x and times t, computes and returns the time derivative
        of the log marginal density.
        '''
        pass

    @abstracthmethod
    def get_score_vt_inner_product(
        self,
        x: TensorType['batch_size', 'data_dim'],
        t: TensorType['batch_size'],
        vt: TensorType['batch_size', 'data_dim']
    ) -> TensorType['batch_size']:
        '''
        Given some values x and times t and vector field vt, computes and returns the
        inner product between the score at time t and vt
        '''
        pass

    @abstracthmethod
    def get_vt_divergence(
        self,
        x: TensorType['batch_size', 'data_dim'],
        t: TensorType['batch_size'],
        vt: TensorType['batch_size', 'data_dim']
    ) -> TensorType['batch_size']:
        '''
        Given some values x and times t and vector field vt, computes and returns the
        divergence of v at time t
        '''
        pass

    def get_extra_terms(
        self,
        x: TensorType['batch_size', 'data_dim'],
        t: TensorType['batch_size'],
        vt: TensorType['batch_size', 'data_dim']
    ) -> TensorType['batch_size']:
        '''
        Given some values x and times t and vector field vt, computes any extra
        terms which should be added to the continuity error.  Could be change in
        normalizing constant over time (i.e., dt log Zt)
        '''
        return 0.0
