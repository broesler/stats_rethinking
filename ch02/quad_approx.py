#!/usr/bin/env python3
#==============================================================================
#     File: grid_approx.py
#  Created: 2019-06-17 11:17
#   Author: Bernie Roesler
#
"""
  Description: Show quadratic approximation improvement with more data.
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

from matplotlib.gridspec import GridSpec
from scipy import stats

plt.style.use('seaborn-darkgrid')
np.random.seed(123)  # initialize random number generator

#------------------------------------------------------------------------------ 
#        Define Parameters
#------------------------------------------------------------------------------
# Data
ks = (6, 12, 24)  # number of event occurrences, i.e. "heads"
ns = (9, 18, 36)  # number of trials, i.e. "tosses"

Nd = len(ks)  # number of times to double data
fig = plt.figure(1, figsize=(12, 5), clear=True)
gs = GridSpec(nrows=1, ncols=Nd)

# Error plots
fig2 = plt.figure(2, figsize=(12, 5), clear=True)
gs2 = GridSpec(nrows=1, ncols=2)
ax21 = fig2.add_subplot(gs2[0])
err = np.empty(Nd)

for i, data in enumerate(zip(ks, ns)):
    k, n = data[0], data[1]
    ## Compute quadratic approximation
    with pm.Model() as normal_approx:
        p = pm.Uniform('p', 0, 1)  # prior distribution of p
        w = pm.Binomial('w', n=n, p=p, observed=k)  # likelihood
        map_est = pm.find_MAP()  # use MAP estimation for mean
        mean_p = map_est['p']  # extract desired value
        std_p = ((1 / pm.find_hessian(map_est, vars=[p]))**0.5)[0,0]

    norm_a = stats.norm(mean_p, std_p)  # quadratic approximation
    beta = stats.beta(k+1, n-k+1)       # analytical posterior

    # Numerical PDFs of each distribution
    p_fine = np.linspace(0, 1, num=100)
    norm_ap = norm_a.pdf(p_fine) 
    beta_p = beta.pdf(p_fine)

    ax = fig.add_subplot(gs[i])  # create the axis

    # Plot the normal approximation
    ax.plot(p_fine, norm_ap,
            # 'C3', label=f'Quad Approx: $\mathcal{{N}}({mean_p:.2f}, {std_p:.2f})$')
            'C3', label=f'Quadratic Approximation')
    ax.axvline(p_fine[norm_ap.argmax()], c='C3', ls='--', lw=1)

    # Plot the analytical posterior
    ax.plot(p_fine, beta_p,
            'k-', label=f'True Posterior')
    ax.axvline(p_fine[beta_p.argmax()], c='k', ls='--', lw=1)

    ax.grid(True)
    ax.set(title=f'trials: {n}, events: {k}',
           xlabel='probability of water, $p$',
           ylabel='non-normalized posterior probability of $p$')

    # Plot the error vs p
    ax21.plot(p_fine, np.abs(norm_ap - beta_p), label=f'N = {n}')
    ax21.set_yscale('log')
    ax21.set(title='Error vs. $p$', 
            xlabel='$p$', 
            ylabel='$\mid \mathcal{N} - \\beta \mid$')

    # point estimate of error for given n
    err[i] = np.linalg.norm(norm_ap - beta_p)

# Error plot of norms
ax22 = fig2.add_subplot(gs2[1])
ax22.plot(ns, err, 'o-')
ax22.set_yscale('log')
ax22.set(title='Error vs. $n$', 
        xlabel='$n$', 
        ylabel='$\Vert \mathcal{N} - \\beta \Vert_2$')

gs2.tight_layout(fig2)

# Plot formatting
gs.tight_layout(fig)
ax.legend(loc=2)
plt.show()

#==============================================================================
#==============================================================================
