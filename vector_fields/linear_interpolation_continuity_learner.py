from continuous_time_db.odes import BaseVectorFieldContinuityLearner

class LinearInterpolationVectorField(BaseVectorFieldContinuityLearner):
    def __init__(
        self,
        reward_function: RewardFunction,
        vector_field: VectorField
        reference_gaussian_sigma: float = 1.0,
        reference_gaussian_norm_p: float = 2.0
        error_norm_ord: float = 2.0
    ):
        super().__init__(error_norm_ord)
        self.reward_fxn = reward_function

        self.energy_predictor = energy_predictor
        self.dt_log_Z_predictor = dt_log_Z_predictor

        def _gaussian_energy(x: TensorType['batch_size', 'data_dim'])['batch_size']:
            return torch.linalg.norm(
                x / reference_gaussian_sigma,
                ord=reference_gaussian_norm_p,
                dim=-1
            ).pow(reference_gaussian_norm_p)

        self.gaussian_energy_fxn = _gaussian_energy

    def get_dt_log_pt(
        self,
        x: TensorType['batch_size', 'data_dim'],
        t: TensorType['batch_size']
    ) -> TensorType['batch_size']:
        '''
        Use energy ft(x) = (1-t)f0(x) + tR(x) + t(1-t)ftheta(x)
        so that
            dt ft = R(x) - f0(x) + (1-2t)ftheta(x, t) + (t - t^2) dt ftheta(x, t)
        '''
        return torch.func.vmap(
            torch.func.grad(self.get_log_pt, argnums=1),
            in_dims=(0, 0)
        )(x, t)

    def get_log_pt(
        self,
        x: TensorType['batch_size', 'data_dim'],
        t: TensorType['batch_size']
    ) -> TensorType['batch_size']:
        '''
        Returns unnormalized log pt with energies
        ft(x) = (1-t)f0(x) + tR(x) + t(1-t)ftheta(x)
        '''
        rewards = t * self.reward_fxn(x)
        ref_energies = (1 - t) * self.gaussian_energy_fxn(x)
        pred_energy = t * (1 - t) * self.energy_predictor(x, t)

        return rewards + ref_energies + pred_energy

    def get_score_vt_inner_product(
        self,
        x: TensorType['batch_size', 'data_dim'],
        t: TensorType['batch_size'],
        vt: TensorType['batch_size', 'data_dim']
    ) -> TensorType['batch_size']:
        score = torch.func.vmap(
            torch.func.grad(self.get_log_pt, argnums=0),
            in_dims=(0, 0)
        )(x, t)

        return (score * vt).sum(dim=-1)

    def get_vt_divergence(
        self,
        x: TensorType['batch_size', 'data_dim'],
        t: TensorType['batch_size'],
        vt: TensorType['batch_size', 'data_dim']
    ) -> TensorType['batch_size']:
        '''
        This is debug stuff for toy setups, so just do closed form
        '''
        jacobians = torch.func.vmap(
            torch.func.grad(self.vector_field, argnums=0),
            in_dims=(0,0)
        )(x, t)

        # Get jacobian trace
        return jacobians.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
