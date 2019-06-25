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
    prior_func : callable, optional, default None
        Function of one parameter describing the prior distribution.
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
    prior = prior_func(p_grid)
    likelihood = stats.binom.pmf(k, n, p_grid)  # binomial distribution
    posterior = likelihood * prior
    if norm_post:
        posterior = posterior / np.sum(posterior)
    return p_grid, posterior, prior_func

#==============================================================================
#==============================================================================
