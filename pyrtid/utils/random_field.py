"""
utility to generate random field in 1, 2 and 3D.

@author: acollet
"""
from typing import Any, Dict, List, Optional, Union

import gstools as gs
import numpy as np

from pyrtid.utils.types import NDArrayFloat

# pylint: disable=C0103  # Do not conform to snake_case naming style
# pylint: disable=R0913  # Too many arguments

# Use the optional dependency GSTools-Core, which is a re-implementation of the main
# algorithms used in GSTools. The new package uses the language Rust and it should be
# faster (in some cases by orders of magnitude), safer, and it will potentially
# completely replace the current standard implementation in Cython.
gs.config.USE_RUST = True


def gen_random_ensemble(
    model: gs.covmodel.CovModel,
    n_ensemble: int,
    var: float,
    len_scale: Union[float, List[float], NDArrayFloat],
    mean: float,
    nx: int,
    ny: int = 1,
    nz: int = 1,
    seed: int = 20170519,
    normalizer: Optional[gs.normalizer.Normalizer] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> NDArrayFloat:
    r"""
    Generate a (ne, nx, ny, nz) ensemble of 3D random fields.

    Parameters
    ----------
    model: CovModel
        Covariance Model class object related to the field.
    n_ensemble : int
        Number of members in the ensemble.
    var : float
        Variance of the model (the nugget is not included in “this” variance).
    len_scale : Union[float, List[float], NDArrayFloat]
        Length scale of the model. If a single value is given, the same length-scale
        will be used for every direction. If multiple values (for main and transversal
        directions) are given, anis will be recalculated accordingly.
        If only two values are given in 3D, the latter one will be used for both
        transversal directions.
    mean: float
        Mean of the field.
    nx : int
        x size of the members.
    ny : int, optional
        y size of the members (if members are 2D). The default is 1 (if members are 1D).
    nz : int, optional
        z size of the members (if members are 3D). The default is 1 (if members are 1D).
    seed : int, optional
        Specifying a seed, we make sure to create reproducible results.
        The default is 20170519.
    normalizer: Optional[Normalized]
        Normalizer to be applied to the field. The default is None.
    model_kwargs: Dict[str, Any]
        additional parameters for the model.

    Returns
    -------
    ens : np.ndarray
        Ensemble members. Dimensions are (ne, nx, ny, nz)

    """
    if model_kwargs is None:
        model_kwargs = {}
    # model = gs.Gaussian(dim=3, var=var, len_scale=len_scale)
    srf = gs.SRF(
        model(dim=3, var=var, len_scale=len_scale, **model_kwargs),
        mean=mean,
        normalizer=normalizer,
    )
    srf.set_pos([range(nx), range(ny), range(nz)], "structured")
    ens = np.zeros((n_ensemble, nx, ny, nz))
    _seed = gs.random.MasterRNG(seed)
    for i in range(n_ensemble):
        srf(seed=_seed(), store=f"better_field{i}")
    for i in range(n_ensemble):
        ens[i] = srf[i]
    return ens


def get_normalized_mean_from_lognormal_params(mean: float, std: float) -> float:
    r"""
    Get the mean of the normalized log-normal distribution.

    Let $X$ be log-normally distributed. Denote $\mu_{X}$ and $\sigma_{X}$ as the mean
    and standard deviation of $X$. The mean $\mu$ of $\log{X}$ is given by:

    .. math::
        \mu = \ln\left(\frac{\mu_X^2}{\sqrt{\mu_X^2+\sigma_X^2}}\right)

    See: https://en.wikipedia.org/wiki/Log-normal_distribution#Generation_and_parameters

    Parameters
    ----------
    mean : float
        The mean of the log-normal distribution.
    std : float
        The standard deviation of the log-normal distribution.

    Returns
    -------
    float
        The mean of the noramlized distribution.
    """
    return np.log(mean**2 / np.sqrt(mean**2 + std**2))


def get_log_normalized_mean_from_normal_params(mean: float, std: float) -> float:
    r"""
    Get the mean of the log-normalized normal distribution.

    Let $Z$ be normally distributed. Denote $\mu_{Z}$ and $\sigma_{Z}$ as the mean and
    standard deviation of $Z$. The mean $\mu$ of $e_{Z}$ is given by:

    .. math::
        \mu = e^{\mu_{Z} + \tfrac{1}{2}\sigma_{Z}^2}


    See: https://en.wikipedia.org/wiki/Log-normal_distribution#Arithmetic_moments

    Parameters
    ----------
    mean : float
        The mean of the normal distribution.
    std : float
        The standard deviation of the normal distribution.

    Returns
    -------
    float
        The mean of the log-noramlized distribution.
    """
    return np.exp(mean + 1 / 2 * std**2)


def get_normalized_std_from_lognormal_params(mean: float, std: float) -> float:
    r"""
    Get the standard deviation of the normalized log-normal distribution.

    Let $X$ be log-normally distributed. Denote $\mu_{X}$ and $\sigma_{X}$ as the mean
    and standard deviation of $X$. The standard deviation $\sigma$ of $\log{X}$ is
    given by:

    .. math::
        \sigma = \sqrt{\ln \left(1+{\frac {\sigma _{X}^{2}}{\mu _{X}^{2}}}\right)}

    See: https://en.wikipedia.org/wiki/Log-normal_distribution#Generation_and_parameters

    Parameters
    ----------
    mean : float
        The mean of the log-normal distribution.
    std : float
        The standard deviation of the log-normal distribution.

    Returns
    -------
    float
        The standard deviation of the normalized distribution.
    """
    return np.sqrt(np.log(1 + std**2 / mean**2))


def get_log_normalized_std_from_normal_params(mean: float, std: float) -> float:
    r"""
    Get the mean of the log-normalized normal distribution.

    Let $Z$ be normally distributed. Denote $\mu_{Z}$ and $\sigma_{Z}$ as the mean and
    standard deviation of $Z$. The standard deviation $\sigma$ of $e_{Z}$ is given by:

    .. math::
        \sigma = e^{\mu + \tfrac{1}{2}\sigma^2}\sqrt{e^{\sigma^2} - 1}


    See: https://en.wikipedia.org/wiki/Log-normal_distribution#Arithmetic_moments

    Parameters
    ----------
    mean : float
        The mean of the normal distribution.
    std : float
        The standard deviation of the normal distribution.

    Returns
    -------
    float
        The standard deviation of the log-normalized distribution.
    """
    return np.exp(mean + 1 / 2 * std**2) * np.sqrt(np.exp(std**2) - 1)
