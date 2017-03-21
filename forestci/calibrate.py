import functools
import scipy.interpolate
import itertools


def neg_loglik(eta, XX):
        g_eta_raw = np.exp(np.dot(XX, eta)) * float(xvals >= 0)
        if (
         (sum(g_eta_raw) == math.inf) |
         (sum(g_eta_raw) <= 100 * np.finfo(np.double).tiny)):
            return (1000 * (length(X) + sum(eta ^ 2)))

        g_eta_main = g_eta_raw / sum(g_eta_raw)
        g_eta = ((1 - unif_fraction) * g_eta_main +
                 unif_fraction * float(xvals >= 0) / sum(xvals >= 0))
        f_eta = np.convolve(g_eta, noise_rotate)

        return np.sum(numpy.interp(xvals,
                      -np.log(maximum(f_eta, 0.0000001)), X))


def gfit(X, sigma, p=2, nbin=1000, unif_fraction=0.1):
    """
    Fit an empirical Bayes prior in the hierarchical model
        mu ~ G, X ~ N(mu, sigma^2)

    Parameters
    ----------
    X: ndarray
        A 1D array of observations
    sigma: float
        noise estimate on X
    p: int
        tuning parameter -- number of parameters used to fit G
    nbin: int
        tuning parameter -- number of bins used for discrete approximation
    unif_fraction: float
        tuning parameter -- fraction of G modeled as "slab"

    Returns
    -------
    An array of the posterior density estimate g

    Notes
    -----
    .. [Efron2014] B Efron. "Two modeling strategies for empirical Bayes
        estimation." Stat. Sci., 29(2): 285–301, 2014.
    """
    min_x = min(min(X) - 2 * np.std(X), 0)
    max_x = max(max(X) + 2 * np.std(X))
    binw = (max_x - min_x) / (nbin - 1)
    xvals = np.arange(min_x, max_x + 1, binw)

    zero_idx = max(np.where(xvals <= 0)[0])
    noise_kernel = norm().pdf(xvals / sigma) * binw / sigma

    if zero_idx > 0:
        noise_rotate = noise_kernel[list(np.arange(zero_idx, len(xvals))) +
                                    list(np.arange(0, zero_idx))]
    else:
        noise_rotate = noise_kernel

    XX = np.zeros((p, len(xvals)), dtype=np.float)
    for ind, exp in enumerate(range(p)):
        mask = np.ones_like(xvals)
        mask[np.where(xvals <= 0)[0]] = 0
        XX[ind, :] = np.pow(xvals, exp) * mask

    eta_hat = minimize(neg_loglik, itertools.repeat(-1, p))
    g_eta_raw = exp(np.dot(XX, eta_hat)) * float(xvals >= 0)
    g_eta_main = g_eta_raw / sum(g_eta_raw)
    g_eta = ((1 - unif_fraction) * g_eta_main +
             unif_fraction * float(xvals >= 0) / sum(xvals >= 0))

    return xvals, g_eta


def gbayes(x0, g_est, sigma):
    """
    Bayes posterior estimation with Gaussian noise

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

    Notes
    -----
     .. [Efron2014] B Efron. "Two modeling strategies for empirical Bayes
          estimation." Stat. Sci., 29(2): 285–301, 2014.
    """

    Kx = norm().pdf((g_est - x0) / sigma)
    post = post / sum(post)
    sum(post * g_est)


def calibrateEB(vars, sigma2):
    """
    Empirical Bayes calibration of noisy variance estimates

    Parameters
    ----------
    vars: ndarray
        list of variance estimates
    sigma2: int
        estimate of the Monte Carlo noise in vars

    Returns
    -------
    An array of the calibrated variance estimates

    Notes
    -----
     .. [Efron2014] B Efron. "Two modeling strategies for empirical Bayes
          estimation." Stat. Sci., 29(2): 285–301, 2014.
    """

    if (sigma2 <= 0 | min(vars) == max(vars)):
        return(np.maximum(vars, 0))
        sigma = np.sqrt(sigma2)
        eb_prior = gfit(vars, sigma)

    if (length(vars >= 200)):
        # If there are many  points use interpolation to speed up computations
        calib_x = quantile(vars, np.arange(0, 1.02, .02))
        calib_y = list(map(gbayes(xx, eb_prior, sigma), calib_x))
        calib_y = map(functools.partial(gbayes, g_est=eb_prior, sigma=sigma),
                      calib_x)
        calib_all = scipy.interpolate.interp1d(calib_x, calib_y)(vars)
    else:
        calib_all = map(functools.partial(gbayes, g_est=eb_prior, sigma=sigma),
                        vars)

    return(calib_all)
