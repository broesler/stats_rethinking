#!/usr/bin/env python3
#==============================================================================
#     File: grid_approx.py
#  Created: 2019-06-17 11:17
#   Author: Bernie Roesler
#
"""
  Description: Grid approximation example.
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec

from scipy.stats import binom

def get_binom_posterior(Np, k, n):
    """Posterior probability given a binomial distribution likelihood and
    uniform prior.

    Parameters
    ----------
    Np : int
        Number of parameter values to use.
    k : int
        Number of event occurrences observed.
    n : int
        Number of trials performed.

    Returns
    -------
    p_grid : (Np, 1) ndarray
        Vector of probability values
    posterior : (Np, 1) ndarray
        Vector of posterior probability values.
    """
    p_grid = np.linspace(0, 1, Np)  # vector of possible parameter values
    prior = np.ones(Np)             # uniform prior
    likelihood = binom.pmf(k, n, p_grid)
    posterior_u = likelihood * prior
    posterior = posterior_u / np.sum(posterior_u)  # standardize to 1
    return p_grid, posterior

k = 6  # number of event occurrences, i.e. "heads"
n = 9  # number of trials, i.e. "tosses"
Nps = [5, 20]

fig = plt.figure(1, clear=True)
plt.suptitle(f'$X \sim B({k}, {n})$')
gs = GridSpec(nrows=1, ncols=2)
for i in range(len(Nps)):
    ax = fig.add_subplot(gs[i])  # left side plot
    p_grid, posterior = get_binom_posterior(Nps[i], k, n)
    ax.plot(p_grid, posterior, 
            marker='o', markerfacecolor='none')
    ax.set_title(f'$N_p$ = {Nps[i]}')
    ax.set_xlabel('actual probability of water')
    ax.set_ylabel('posterior probability')
    ax.grid()

plt.tight_layout()
plt.show()

#==============================================================================
#==============================================================================
