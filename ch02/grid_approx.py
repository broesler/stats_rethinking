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
import pymc3 as pm

from matplotlib.gridspec import GridSpec
from scipy.stats import binom

# Possible prior distributions
PRIOR_D = dict({'uniform': dict({'prior': lambda p: np.ones(p.shape),
                                 'title': f'$U(0, 1)$'}),
                'step': dict({'prior': lambda p: np.where(p < 0.5, 0, 1),
                              'title': '0 where $p < 0.5$, 1 otherwise'}),
                'exp': dict({'prior': lambda p: np.exp(-5 * np.abs(p - 0.5)),
                             'title': '$-5e^{{|p - 0.5|}}$'})
                })


def get_binom_posterior(Np, k, n, prior_str='uniform'):
    """Posterior probability given a binomial distribution likelihood and
    arbitrary prior.

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
    prior = PRIOR_D[prior_str]['prior'](p_grid)
    likelihood = binom.pmf(k, n, p_grid)
    posterior_u = likelihood * prior
    posterior = posterior_u / np.sum(posterior_u)  # standardize to 1
    return p_grid, posterior


prior_str = 'uniform'  # 'uniform', 'step', 'exp'
k = 6  # number of event occurrences, i.e. "heads"
n = 9  # number of trials, i.e. "tosses"

Nps = [5, 20]  # range of grid sizes to try
NN = len(Nps)

# Compute quadratic approximation

#------------------------------------------------------------------------------ 
#        Plot Results
#------------------------------------------------------------------------------
fig = plt.figure(1, clear=True)
gs = GridSpec(nrows=1, ncols=NN)

for i in range(NN):
    Np = Nps[i]

    p_grid, posterior = get_binom_posterior(Np, k, n, prior_str=prior_str)
    p_max = p_grid[np.where(posterior == np.max(posterior))]
    p_max = p_max.mean() if p_max.size > 1 else p_max.item()

    # Plot the result
    ax = fig.add_subplot(gs[i])  # left side plot
    ax.axvline(p_max, ls='--', c='k', lw=1)
    ax.plot(p_grid, posterior, 
            marker='o', markerfacecolor='none', label='Posterior')
    # ax.plot(p_grid, prior, 'k-', label='prior')

    ax.set_title(f'$N_p$ = {Np}, $p_{{max}}$ = {p_max:0.2f}')
    ax.set_xlabel('probability of water')
    ax.set_ylabel('posterior probability')
    ax.grid()

base_title = '$X \sim $ {} |  events: {}, trials: {}'
fig.suptitle(base_title.format(PRIOR_D[prior_str]['title'], k, n))
fig.subplots_adjust(top=0.2)
gs.tight_layout(fig)
plt.show()

#==============================================================================
#==============================================================================
