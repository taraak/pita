from torch import vmap
import torch 
from torch.func import hessian
from scipy.stats import qmc
import numpy as np


def sample_from_tensor(tensor, num_samples):
    idx = torch.randint(0, tensor.shape[0], (num_samples,))
    return tensor[idx]

def rademacher(shape, dtype=torch.float32, device='cuda'):
    """Sample from Rademacher distribution."""
    rand = ((torch.rand(shape) < 0.5)) * 2 - 1
    return rand.to(dtype).to(device)

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
    jvp = torch.autograd.grad(nabla_Ut, xt, noise, create_graph=True)[0] # nabla(nabla_Ut) * noise
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
    u = (u + 1/bs*torch.arange(bs)) % 1.0
    clipped_weights = torch.clip(torch.softmax(logits, dim=-1), 1e-6, 1.0)
    # clipped_weights = torch.softmax(logits, dim=-1)
    bins = torch.cumsum(clipped_weights, dim=-1)
    ids = np.digitize(u, bins.cpu(), right=True)

    ids[ids == logits.shape[-1]] = ids[ids == logits.shape[-1]] - 1 
    return ids, None

# ito density estimator for derivative of log density
def ito_logdensity(sde, t, x, dt):
    if t.dim() == 0:
        # repeat the same time for all points if we have a scalar time
        t = t * torch.ones(x.shape[0]).to(x.device)
    # return ((nabla_qt * dx).sum(-1) 
    #         - 0.5 * (sde.g(t, x)[:, None] * nabla_qt).pow(2).sum(-1) * dt).detach()
    dwt = sde.noise * np.sqrt(dt)
    return (sde.g(t, x) * (sde.nabla_logqt * dwt).sum(-1) 
            + 0.5 * (sde.g(t, x)[:, None] * sde.nabla_logqt).pow(2).sum(-1) * dt)
