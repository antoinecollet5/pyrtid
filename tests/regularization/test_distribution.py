import numpy as np
import pytest
from pyrtid.regularization import ProbDistFitting
from pyrtid.regularization.distribution import (
    cdf_distance,
    cdf_distance_gradient,
    get_cdfs,
)
from pyrtid.utils import NDArrayFloat
from pyrtid.utils.finite_differences import is_gradient_correct


@pytest.mark.parametrize("p", (1, 2, 3))
def test_prob_dist_fitting(p) -> None:
    multimodal_dist_target = np.concatenate(
        (
            np.random.normal(10, 3, 1000),
            np.random.normal(30, 5, 4000),
            np.random.normal(45, 6, 500),
        ),
        axis=0,
    )

    multimodal_dist_init = np.concatenate(
        (
            np.random.normal(-10, 3, 50),
            np.random.normal(20, 5, 200),
            np.random.normal(60, 6, 25),
        ),
        axis=0,
    )

    multimodal_dist_target = np.concatenate(
        (
            multimodal_dist_target,
            multimodal_dist_target[:500],
            multimodal_dist_target[:250],
        )
    )

    multimodal_dist_init = np.concatenate(
        (multimodal_dist_init, multimodal_dist_init[:50], multimodal_dist_init[:25])
    )

    dist_range = (
        min(np.min(multimodal_dist_target).item(), np.min(multimodal_dist_init).item()),
        max(np.max(multimodal_dist_target).item(), np.max(multimodal_dist_init).item()),
    )

    pv, bins_target = np.histogram(multimodal_dist_target, bins=50, range=dist_range)
    v = (bins_target[:-1] + bins_target[1:]) / 2

    pu, bins_init = np.histogram(multimodal_dist_init, bins=50, range=dist_range)
    u = (bins_init[:-1] + bins_init[1:]) / 2

    assert cdf_distance(p, v, v, pv, pv) < 1e-10
    assert cdf_distance(p, u, u, pu, pu) < 1e-10
    assert cdf_distance(p, u, v, pu, pv) > 0.5

    get_cdfs(u, v)  # to test the absence of weights

    reg = ProbDistFitting(
        target_values=v,
        target_weights=pv,
        order=p,
        sub_selection=np.arange(v.size)[:-10],
    )
    reg.eval_loss(multimodal_dist_init)

    assert is_gradient_correct(
        multimodal_dist_init, fm=reg.eval_loss, grad=reg.eval_loss_gradient_analytical
    )

    reg = ProbDistFitting(target_values=v, target_weights=pv, order=p)
    reg.eval_loss(multimodal_dist_init)

    weights = np.abs(np.random.normal(loc=10, size=multimodal_dist_init.size))

    def cost_fun(x) -> float:
        return cdf_distance(p, x, multimodal_dist_target, weights) * 1e3

    def cost_fun_grad(x) -> NDArrayFloat:
        return cdf_distance_gradient(p, x, multimodal_dist_target, weights, None) * 1e3

    assert is_gradient_correct(
        multimodal_dist_init, fm=cost_fun, grad=cost_fun_grad, eps=1e-3
    )
