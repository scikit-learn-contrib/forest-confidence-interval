"""
Calibration based on empirical Bayes estimation [Efron2014]_.

This calibration procedure can be useful when the number of trees in the
random forest is small.

"""
import warnings
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


def gfit(X, sigma, p=2, nbin=1000, unif_fraction=0.1):
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
        Number of parameters used to fit G.
    nbin: int
        Number of bins used for discrete approximation.
    unif_fraction: float
        Fraction of G modeled as "slab".

    Returns
    -------
    An array of the posterior density estimate g.
    """
    min_x = min(np.min(X) - 2 * np.std(X, ddof=1), 0)
    max_x = max(max(X) + 2 * np.std(X, ddof=1),
                np.std(X, ddof=1))
    xvals = np.linspace(min_x, max_x, nbin)

    mask = xvals >= 0
    if not np.any(xvals > 0):
        g_eta = np.zeros_like(xvals)
        g_eta[mask] = 1.0 / np.sum(mask)
        return xvals, g_eta

    noise_kernel = norm(scale=sigma,loc=xvals.mean()).pdf(xvals)
    noise_kernel /= noise_kernel.sum()

    assert sum(mask) > 0
    g_eta_slab = mask / sum(mask)

    XX = np.column_stack([ pow(xvals, exp) for exp in range(1, p+1)])
    signed_scale = np.sum(XX, axis=0, keepdims=True)
    absolute_scale = np.sum(np.abs(XX), axis=0, keepdims=True)
    near_zero = (
        np.abs(signed_scale)
        <= np.sqrt(np.finfo(float).eps) * absolute_scale
    )
    scale = np.where(near_zero, absolute_scale, signed_scale)
    XX /= scale
    XX[~mask] = 0.0

    def prior_from_eta(eta):
        log_weights = np.dot(XX[mask], eta)
        if not np.all(np.isfinite(log_weights)):
            return None

        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            return None

        g_eta_main = np.zeros_like(xvals)
        g_eta_main[mask] = weights / weight_sum
        return (
            (1 - unif_fraction) * g_eta_main
            + unif_fraction * g_eta_slab
        )

    def neg_loglik(eta):
        g_eta = prior_from_eta(eta)
        if g_eta is None:
            return 1000 * (len(X) + sum(eta ** 2))

        f_eta = fftconvolve(g_eta, noise_kernel, mode='same')
        return np.sum(np.interp(X, xvals,
                      -np.log(np.maximum(f_eta, 0.0000001))))

    initial_eta = np.full(p, -1, dtype='float')
    res = minimize(
        neg_loglik,
        initial_eta,
        tol=5e-5
    )
    if not res.success:
        fallback = minimize(
            neg_loglik,
            initial_eta,
            method="Nelder-Mead",
            tol=5e-5,
            options={"maxiter": 2000, "maxfev": 4000},
        )
        if fallback.success or (
            np.isfinite(fallback.fun) and fallback.fun < res.fun
        ):
            res = fallback
    if not res.success:
        warnings.warn("Fitting the empirical bayes prior failed with message %s." % res.message)
    eta_hat = res.x
    g_eta = prior_from_eta(eta_hat)

    assert g_eta is not None, "Fitting the empirical bayes prior failed."
    return xvals, g_eta


def gbayes(x0, g_est, sigma):
    """
    Estimate Bayes posterior with Gaussian noise [Efron2014]_.

    Parameters
    ----------
    x0: ndarray
        an observation
    g_est: (ndarray,ndarray)
        a prior density, as returned by gfit
        g_est[0] is the x-positions
        g_est[1] is the densities
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

    if len(variances) >= 200:
        # Interpolate to speed up computations:
        calib_x = np.percentile(variances,
                                np.arange(0, 102, 2))
        calib_y = [gbayes(x,g_est=eb_prior,sigma=sigma) for x in calib_x]
        calib_all = np.interp(variances, calib_x, calib_y)
    else:
        calib_all = [gbayes(x,g_est=eb_prior,sigma=sigma) for x in variances]

    return np.asarray(calib_all)
