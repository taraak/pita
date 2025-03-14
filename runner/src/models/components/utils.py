import numpy as np
import torch
from scipy.stats import qmc
from torch import vmap
from torch.func import hessian


def sample_from_tensor(tensor, num_samples):
    idx = torch.randint(0, tensor.shape[0], (num_samples,))
    return tensor[idx]


def rademacher(shape, dtype=torch.float32, device="cuda"):
    """Sample from Rademacher distribution."""
    rand = (torch.rand(shape) < 0.5) * 2 - 1
    return rand.to(dtype).to(device)


def compute_divergence_forloop(nabla_Ut, x):
    N, d = x.shape
    hessian_matrix = torch.zeros(N, d, d, device=x.device, dtype=x.dtype)

    # Compute the Hessian row-wise
    for i in range(d):  # ∂(∇U_t)_i / ∂x
        grad2 = torch.autograd.grad(
            nabla_Ut[:, i].sum(), x, retain_graph=True, create_graph=True
        )[0]
        hessian_matrix[:, i, :] = grad2  # i-th row
    laplacian = hessian_matrix.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    return laplacian


def compute_laplacian_exact(model, t, xt):
    def func_wrap(t, xt):
        return model(t.unsqueeze(0), xt.unsqueeze(0)).squeeze()

    # Calculate the Hessian matrix of the model output with respect to the input
    hessian_matrix = vmap(hessian(func_wrap, argnums=1))(t, xt)

    # Calculate the Laplacian as the trace of the Hessian matrix
    laplacian = hessian_matrix.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    return laplacian.detach()


def compute_laplacian_hutchinson(nabla_Ut, t, xt):
    # Compute the noise
    noise = rademacher(xt.shape, dtype=torch.float32, device=xt.device)
    # Compute the jvp of the nabla_Ut and the noise
    jvp = torch.autograd.grad(nabla_Ut, xt, noise, create_graph=True)[0]  # nabla(nabla_Ut) * noise
    laplacian = (jvp * noise).sum(-1)
    return laplacian.detach()


def compute_laplacian(model, nabla_Ut, t, xt, n_samples=1, exact=True):
    if exact:
        return compute_laplacian_exact(model, t, xt)
    else:
        laplacian = 0
        for _ in range(n_samples):
            laplacian += compute_laplacian_hutchinson(nabla_Ut, t, xt)
        return laplacian / n_samples


sampler = qmc.Sobol(d=1, scramble=False)


def sample_cat(bs, next_u, logits):
    # u, next_u = sample_uniform(bs, next_u)
    u = sampler.random(bs).squeeze()
    clipped_weights = torch.clip(torch.softmax(logits, dim=-1), 1e-6, 1.0)
    bins = torch.cumsum(clipped_weights, dim=-1)
    ids = np.digitize(u, bins.cpu())
    return ids, next_u


def sample_cat_sys(bs, logits):
    u = torch.rand(size=(1,), dtype=torch.float64)
    u = (u + 1 / bs * torch.arange(bs)) % 1.0
    clipped_weights = torch.clip(torch.softmax(logits, dim=-1), 1e-6, 1.0)
    # clipped_weights = torch.softmax(logits, dim=-1)
    bins = torch.cumsum(clipped_weights, dim=-1)
    ids = np.digitize(u, bins.cpu(), right=True)

    ids[ids == logits.shape[-1]] = ids[ids == logits.shape[-1]] - 1
    return ids, None


def sample_birth_death_clocks(
    bs, accum_birth, accum_death, thresh_times, reset_transition_per_index=True
):
    device = accum_birth.device
    # Generate random keys
    death_mask = accum_death >= thresh_times
    ids = torch.arange(bs).to(device)

    # Sample candidate replacement indices according to accumulated birth weights
    if reset_transition_per_index:
        transition_probs = accum_birth / torch.sum(accum_birth, dim=-1, keepdim=True)
        transition_probs = torch.nan_to_num(transition_probs, nan=0.0)

        row_sums = torch.sum(transition_probs, dim=-1, keepdim=True)
        zero_mask = row_sums == 0.0

        # Replace zero-probability rows with uniform probabilities
        uniform_probs = torch.ones_like(transition_probs) / transition_probs.size(-1)
        transition_probs = torch.where(zero_mask, uniform_probs, transition_probs)
        replace_ids = torch.vmap(lambda x: torch.multinomial(x, 1), randomness="different")(
            transition_probs
        ).squeeze()
    else:
        transition_probs = accum_birth / torch.sum(accum_birth)
        replace_ids = torch.multinomial(transition_probs, bs, replacement=True)

    # Replace those entries chosen for killing
    ids = torch.where(death_mask, replace_ids, ids)

    # Sample new jump thresholds
    new_thresh_times = torch.distributions.Exponential(1.0).sample((bs,)).to(device)
    thresh_times = torch.where(death_mask, new_thresh_times, thresh_times)

    # Reset birth and death weights in killed indices
    if reset_transition_per_index:
        # import pdb; pdb.set_trace()
        accum_birth = torch.where(
            death_mask.unsqueeze(-1), torch.zeros_like(accum_birth), accum_birth
        )
    else:
        accum_birth = torch.where(death_mask, torch.zeros_like(accum_birth), accum_birth)

    accum_death = torch.where(death_mask, torch.zeros_like(accum_death), accum_death)

    metrics = (torch.sum(death_mask),)
    return ids, accum_birth.detach(), accum_death.detach(), thresh_times.detach(), metrics


# ito density estimator for derivative of log density
def ito_logdensity(sde, t, x, dt):
    if t.dim() == 0:
        # repeat the same time for all points if we have a scalar time
        t = t * torch.ones(x.shape[0]).to(x.device)
    dwt = sde.noise * np.sqrt(dt)
    return (
        sde.g(t, x) * (sde.nabla_logqt * dwt).sum(-1)
        + 0.5 * (sde.g(t, x)[:, None] * sde.nabla_logqt).pow(2).sum(-1) * dt
    ).detach()
