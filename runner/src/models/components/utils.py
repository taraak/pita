from torch import vmap
import torch 
from torch.func import hessian

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