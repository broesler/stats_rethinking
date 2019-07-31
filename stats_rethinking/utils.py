#!/usr/bin/env python3
#==============================================================================
#     File: utils.py
#  Created: 2019-06-24 21:35
#   Author: Bernie Roesler
#
"""
  Description: Utility functions for Statistical Rethinking code.
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from scipy import stats
from sklearn.utils.extmath import cartesian


def annotate(text, ax, loc='upper left'):
    """Add annotation `text` to upper left of `ax` frame."""
    # TODO validate loc string first
    # except KeyError:
    #     raise KeyError(f'The location {loc} is not supported!')

    yloc, xloc = loc.split()
    XS = dict({ 'left': 0.05, 'center': 0.5, 'right': 0.95})
    YS = dict({'lower': 0.05, 'center': 0.5, 'upper': 0.95})
    YA = dict({'upper': 'top', 'center': 'center', 'lower': 'bottom'})
    xc, yc, va = XS[xloc], YS[yloc], YA[yloc]
    ax.text(x=xc, y=yc, s=text, ha=xloc, va=va, transform=ax.transAxes)


def quantile(data, q=0.89, width=10, precision=8,
                 q_func=np.quantile, verbose=False, **kwargs):
    """Pretty-print the desired quantile values from the data.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    width : int, optional, default=10
        Width of printing field.
    precision : int, optional, default=8
        Number of decimal places to print.
    q_func : callable, optional, default=numpy.quantile
        Function to compute the quantile outputs from the data.
    verbose : bool, optional, default=True
        Print the output quantile percentages names and values.
    **kwargs
        Additional arguments to `q_func`.

    Returns
    -------
    quantile : scalary or ndarray
        The requested quantiles. See documentation for `numpy.quantile`.

    See Also
    --------
    `numpy.quantile`
    """
    q = np.atleast_1d(q)
    quantiles = q_func(data, q, **kwargs)
    if verbose:
        fstr = f"{width}.{precision}f"
        name_str = ' '.join([f"{100*p:{width-1}g}%" for p in q])
        value_str = ' '.join([f"{q:{fstr}}" for q in quantiles])
        print(f"{name_str}\n{value_str}")
    return quantiles


def percentiles(data, q=50, **kwargs):
    """Pretty-print the desired percentile values from the data.

    ..note:: A wrapper around `quantile`, where the arguments are forced
        to take the form:
    ..math:: a = \frac{1 - q}{2}
        and called with :math:\mathtt{quantile(data, (a, 1-a))}

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 100, inclusive.
    **kwargs
        See `quantile` for additional options.

    See Also
    --------
    `quantile`
    """
    a = (1 - (q/100)) / 2
    quantiles = quantile(data, (a, 1-a), **kwargs)
    return quantiles


def hpdi(data, alpha=0.5, verbose=False, width=10, precision=8, **kwargs):
    """Compute highest probability density interval.

    ..note::
        This function calls `sts.quantile` with `pymc3.stats.hpd` function.
    """
    # subtract from 1, so "alpha=0.89" gives 89% of probability
    # q_arr = 1 - np.atleast_1d(alpha)
    # out = np.empty((q_arr.shape[0], 2))
    # for i, q in enumerate(q_arr):
        # out[i,:] = quantile(data, v, q_func=pm.stats.hpd, **kwargs)
    q = 1 - alpha
    quantiles = pm.stats.hpd(data, q, **kwargs)
    if verbose:
        fstr = f"{width}.{precision}f"
        name_str = ' '.join([f"{100*(1-p):{width-1}g}%" for p in np.hstack((q, q))])
        value_str = ' '.join([f"{q:{fstr}}" for q in quantiles])
        print(f"|{name_str}|\n{value_str}")
    return quantiles


def grid_binom_posterior(Np, k, n, prior_func=None, norm_post=True):
    """Posterior probability assuming a binomial distribution likelihood and
    arbitrary prior.

    Parameters
    ----------
    Np : int
        Number of parameter values to use.
    k : int
        Number of event occurrences observed.
    n : int
        Number of trials performed.
    prior_func : callable, optional, default U(0, 1)
        Function of one parameter describing the prior distribution.
        If prior_func is None, it defaults to a uniform prior
    norm_post : bool, optional, default True
        If True, normalize posterior to a maximum value of 1.

    Returns
    -------
    p_grid : (Np, 1) ndarray
        Vector of parameter values.
    posterior : (Np, 1) ndarray
        Vector of posterior probability values.
    """
    p_grid = np.linspace(0, 1, Np)  # vector of possible parameter values
    if prior_func is None:
        prior = np.ones(Np)  # default uniform prior
    else:
        prior = prior_func(p_grid)
    likelihood = stats.binom.pmf(k=k, n=n, p=p_grid)  # binomial distribution
    posterior = likelihood * prior
    if norm_post:
        posterior = posterior / np.sum(posterior)
    return p_grid, posterior, prior


def density(data, adjust=1.0, **kwargs):
    """Return the kernel density estimate of the data, consistent with
    R function of the same name.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in K dimensions.
    adjust : float, optional, default=1.0
        Multiplicative factor for the bandwidth.
    **kwargs : optional
        Additional arguments passed to `scipy.stats.gaussian_kde`. 

    ..note:: This function overrides the `bw_method` argument. The
      stats_rethinking "dens" (R code 2.9) function calls the following
      R function:
          thed <- density(data, adjust=0.5)
      The default bandwidth in `density` (R docs) is: `bw="nrd0"`, which
      corresponds to 'silverman' in python. `adjust` sets `bandwith *= adjust`.

    Returns
    -------
    kde : kernel density estimate object
        Call kde.pdf(x) to get the actual samples

    """
    kde = stats.gaussian_kde(data, **kwargs)
    kde.set_bandwidth(adjust * kde.silverman_factor())
    return kde


# TODO expand documentation with examples
def expand_grid(**kwargs):
    """Return a DataFrame of points, where the columns are kwargs."""
    return pd.DataFrame(cartesian(kwargs.values()), columns=kwargs.keys())

# TODO:
#     * allow for numpy array of data (change ppf calls to quantile)
#     * allow for DataFrame of samples (cols are variable names)
#     * allow for non-dict (just data) input by excluding "index"
#     parameter from DataFrame call.
def precis(quap, p=0.89):
    """Return a `DataFrame` of the mean, standard deviation, and percentile
    interval of the given `rv_frozen` distributions.
    """
    a = (1-p)/2
    pp = 100*np.array([a, 1-a])  # percentages for printing

    # dictionary of `rv_frozen` distributions
    if isinstance(quap, dict):
        index = quap.keys()
        vals = np.empty((len(quap), 4))
        for i, v in enumerate(quap.values()):
            vals[i,:] = [v.mean(), v.std(), v.ppf(a), v.ppf(1-a)]
        df = pd.DataFrame(vals, index=index,
                          columns=['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"])
        return df

    # DataFrame of data points
    if isinstance(quap, pd.DataFrame):
        index = quap.keys()
        df = pd.DataFrame()
        df['mean'] = quap.mean()
        df['std'] = quap.std()
        for i in range(2):
            df[f"{pp[i]:g}%"] = quap.apply(lambda x: np.percentile(x, pp[i]))
        return df

    # Numpy array of data points
    if isinstance(quap, np.ndarray):
        # Columns are data, ignore index
        vals = np.vstack([quap.mean(axis=0),
                          quap.std(axis=0),
                          np.percentile(quap, pp[0], axis=0),
                          np.percentile(quap, pp[1], axis=0)]).T
        df = pd.DataFrame(vals,
                          columns=['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"])
        return df
    else:
        raise TypeError('quap of this type is unsupported!')


# TODO expand documentation with examples
def quap(varnames, start=None):
    """Return quadratic approximation for the MAP estimate of each variable in
    `varnames`. Must be called within a pymc3 context block.
    """
    pm.sample()  # initialize NUTS sampler
    map_est = pm.find_MAP(start=start)  # use MAP estimation for mean

    quap = dict()
    for k, v in varnames.items():
        mean = map_est[k]
        std = ((1 / pm.find_hessian(map_est, vars=[v]))**0.5)[0,0]
        quap[k] = stats.norm(mean, std)
    return quap


def sample_quap(quap, N=1000):
    """Sample each distribution in the `quap` dictionary."""
    return pd.DataFrame(np.array([v.rvs(N) for v in quap.values()]).T,
                        columns=quap.keys())

#==============================================================================
#==============================================================================
