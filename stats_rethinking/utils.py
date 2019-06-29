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

import numpy as np
from scipy import stats
import pymc3 as pm


def annotate(text, ax):
    """Add annotation `text` to top-left of `ax` frame."""
    ax.text(x=0.05, y=0.9, s=text,
            ha='left',
            va='center',
            transform=ax.transAxes)


def get_quantile(data, q=0.89, width=10, precision=8,
                 q_func=np.quantile, verbose=True, **kwargs):
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


def get_percentiles(data, q=0.5, **kwargs):
    """Pretty-print the desired percentile values from the data.

    ..note:: A wrapper around `get_quantile`, where the arguments are forced
        to take the form:
    ..math:: a = \frac{1 - q}{2}
        and called with :math:\mathtt{get_quantile(data, (a, 1-a))}

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    **kwargs
        See `get_quantile` for additional options.

    See Also
    --------
    `get_quantile`
    """
    a = (1 - q) / 2
    quantiles = get_quantile(data, (a, 1-a), **kwargs)
    return quantiles


def get_hpdi(data, q=0.5, **kwargs):
    """Call `sts.get_quantile` with `pymc3.stats.hpd` function."""
    return get_quantile(data, q, q_func=pm.stats.hpd, **kwargs)


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
    likelihood = stats.binom.pmf(k, n, p_grid)  # binomial distribution
    posterior = likelihood * prior
    if norm_post:
        posterior = posterior / np.sum(posterior)
    return p_grid, posterior, prior_func


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
        Additional arguments passed to `scipy.stats.gaussian_kde`

    Returns
    -------
    kde : kernel density estimate object
        Call kde.pdf(x) to get the actual samples

    ..note:: The stats_rethinking "dens" (R code 2.9) function calls the
      following R function:
          thed <- density(data, adjust=0.5)
      The default bandwidth in `density` (R docs) is: `bw="nrd0"`, which
      corresponds to 'silverman' in python. `adjust` sets `bandwith *= adjust`.
    """
    kde = stats.gaussian_kde(data)
    kde.set_bandwidth(adjust * kde.silverman_factor())
    return kde

#==============================================================================
#==============================================================================
