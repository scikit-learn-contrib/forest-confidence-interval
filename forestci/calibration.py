"""
Calibration based on empirical Bayes estimation [Efron2014]_.

This calibration procedure can be useful when the number of trees in the
random forest is small.

"""
import functools
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.stats import norm
from .due import _due, _BibTeX

__all__ = ("gfit", "gbayes", "calibrateEB")


_due.cite(_BibTeX("""
@ARTICLE{Wager2014-wn,
  title       = "Two modeling strategies for empirical Bayes estimation.",
  author      = Efron, Bradley
  journal     = "Stat. Sci.",
  volume      =  29,
  number      =  2,
  pages       = "285--301",
  month       =  feb,
  year        =  2014,}"""),
          description=("Confidence Intervals for Random",
          " Forests: The Jackknife and the Infinitesimal",
                       "Jackknife"),
          path='forestci')


def gfit(X, sigma, p=5, nbin=200, unif_fraction=0.1):
    """
    Fit empirical Bayes prior in the hierarchical model [Efron2014]_.

    .. math::

        mu ~ G, X ~ N(mu, sigma^2)

    Parameters
    ----------
    X: ndarray
        A 1D array of observations.
    sigma: float
        Noise estimate on X.
    p: int
        Number of parameters used to fit G. Default: 5
    nbin: int
        Number of bins used for discrete approximation.
        Default: 200
    unif_fraction: float
        Fraction of G modeled as "slab". Default: 0.1

    Returns
    -------
    An array of the posterior density estimate g.
    """
    min_x = min(min(X) - 2 * np.std(X, ddof=1), 0)
    max_x = max(max(X) + 2 * np.std(X, ddof=1),
                np.std(X, ddof=1))
    xvals = np.linspace(min_x, max_x, nbin)
    binw = (max_x - min_x) / (nbin - 1)

    zero_idx = max(np.where(xvals <= 0)[0])
    noise_kernel = norm().pdf(xvals / sigma) * binw / sigma

    if zero_idx > 0:
        noise_rotate = noise_kernel[list(np.arange(zero_idx, len(xvals))) +
                                    list(np.arange(0, zero_idx))]
    else:
        noise_rotate = noise_kernel

    XX = np.zeros((p, len(xvals)), dtype=np.float)
    for ind, exp in enumerate(range(1, p+1)):
        mask = np.ones_like(xvals)
        mask[np.where(xvals <= 0)[0]] = 0
        XX[ind, :] = pow(xvals, exp) * mask
    XX = XX.T

    def neg_loglik(eta):
        mask = np.ones_like(xvals)
        mask[np.where(xvals <= 0)[0]] = 0
        g_eta_raw = np.exp(np.dot(XX, eta)) * mask
        if ((np.sum(g_eta_raw) == np.inf) |
            (np.sum(g_eta_raw) <=
                100 * np.finfo(np.double).tiny)):
                return (1000 * (len(X) + sum(eta ** 2)))

        g_eta_main = g_eta_raw / sum(g_eta_raw)
        g_eta = ((1 - unif_fraction) * g_eta_main +
                 unif_fraction * mask / sum(mask))
        f_eta = fftconvolve(g_eta, noise_rotate, mode='same')
        return np.sum(np.interp(X, xvals,
                      -np.log(np.maximum(f_eta, 0.0000001))))

    eta_hat = minimize(neg_loglik,
                       list(itertools.repeat(-1, p))).x
    g_eta_raw = np.exp(np.dot(XX, eta_hat)) * mask
    g_eta_main = g_eta_raw / sum(g_eta_raw)
    g_eta = ((1 - unif_fraction) * g_eta_main +
             unif_fraction * mask) / sum(mask)

    return xvals, g_eta


def gbayes(x0, g_est, sigma):
    """
    Estimate Bayes posterior with Gaussian noise [Efron2014]_.

    Parameters
    ----------
    x0: ndarray
        an observation
    g_est: float
        a prior density, as returned by gfit
    sigma: int
        noise estimate

    Returns
    -------
    An array of the posterior estimate E[mu | x0]
    """

    Kx = norm().pdf((g_est[0] - x0) / sigma)
    post = Kx * g_est[1]
    post /= sum(post)
    return sum(post * g_est[0])


def calibrateEB(variances, sigma2):
    """
    Calibrate noisy variance estimates with empirical Bayes.

    Parameters
    ----------
    vars: ndarray
        List of variance estimates.
    sigma2: int
        Estimate of the Monte Carlo noise in vars.

    Returns
    -------
    An array of the calibrated variance estimates
    """
    if (sigma2 <= 0 or min(variances) == max(variances)):
        return(np.maximum(variances, 0))
    sigma = np.sqrt(sigma2)
    eb_prior = gfit(variances, sigma)
    # Set up a partial execution of the function
    part = functools.partial(gbayes, g_est=eb_prior,
                             sigma=sigma)
    if len(variances) >= 200:
        # Interpolate to speed up computations:
        calib_x = np.percentile(variances,
                                np.arange(0, 102, 2))
        calib_y = list(map(part, calib_x))
        calib_all = np.interp(variances, calib_x, calib_y)
    else:
        calib_all = list(map(part, variances))

    return np.asarray(calib_all)
