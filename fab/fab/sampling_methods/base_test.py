from functools import partial

import functorch
import torch
import torch.nn as nn
from experiments.make_flow.make_normflow_model import make_wrapped_normflow_realnvp

from fab.sampling_methods.base import (
    Point,
    create_point,
    get_grad_intermediate_log_prob,
    get_intermediate_log_prob,
)
from fab.target_distributions.gmm import GMM


def test_create_point():
    dim = 2
    batch_size = 10
    flow = make_wrapped_normflow_realnvp(dim=dim)
    target = GMM(dim, n_mixes=3, loc_scaling=1)
    x = torch.randn((batch_size, dim))
    point = create_point(x=x, log_q_fn=flow.log_prob, log_p_fn=target.log_prob, with_grad=True)
    print(point)


if __name__ == "__main__":
    test_create_point()
