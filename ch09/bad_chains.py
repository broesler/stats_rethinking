#!/usr/bin/env python3
# =============================================================================
#     File: bad_chains.py
#  Created: 2023-12-04 15:45
#   Author: Bernie Roesler
#
"""
§9.5 Poor Monte Carlo chains.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

obs = np.r_[-1, 1]

with pm.Model():
    α = pm.Normal('α', 0, 1000)
    μ = pm.Deterministic('μ', α)
    σ = pm.Exponential('σ', 0.0001)
    y = pm.Normal('y', μ, σ, observed=obs)
    m9_2 = sts.ulam(chains=2)

print('m9.2:')
sts.precis(m9_2)

# m9_2.plot_trace(title='m9.2')
# m9_2.pairplot(title='m9.2')

with pm.Model():
    α = pm.Normal('α', 0, 10)
    μ = pm.Deterministic('μ', α)
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=obs)
    m9_3 = sts.ulam(chains=2)

print('m9.3:')
sts.precis(m9_3)

# m9_3.plot_trace(title='m9.3')
# m9_3.pairplot(title='m9.3')

# ----------------------------------------------------------------------------- 
#         Plot Figure 9.10
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(num=1, ncols=2, clear=True)

prior = m9_3.sample_prior(10_000)


def plot_densities(ax, v, xlim=None, xlabel=None):
    # Sample posterior
    vals = m9_3.samples[v].stack(sample=('chain', 'draw'))
    vals = np.sort(vals)
    dens = stats.gaussian_kde(vals).pdf(vals)
    ax.plot(vals, dens, c='C0', label='posterior')

    # Evaluate the prior probability directly using the model
    vals = np.linspace(*xlim)
    dens = np.exp(pm.logp(m9_3.model[v], vals).eval())
    ax.plot(vals, dens, ls='--', c='k', label='prior')

    ax.set(xlabel=xlabel or v,
           ylabel='Density',
           xlim=xlim,
           ylim=(0., None))

    ax.spines[['top', 'right']].set_visible(False)

    return ax


plot_densities(
    axs[0],
	'α',
    xlim=(-15, 15),
	xlabel=r'$\alpha$'
)

plot_densities(
    axs[1],
	'σ',
    xlim=(0, 10),
	xlabel=r'$\sigma$'
)

axs[0].legend()


# ----------------------------------------------------------------------------- 
#         Non-identifiable parameters
# -----------------------------------------------------------------------------
obs = stats.norm.rvs(100)

with pm.Model():
    α0 = pm.Normal('α0', 0, 1000)
    α1 = pm.Normal('α1', 0, 1000)
    μ = pm.Deterministic('μ', α0 + α1)
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=obs)
    m9_4 = sts.ulam()

print('m9.4:')
sts.precis(m9_4)

with pm.Model():
    α0 = pm.Normal('α0', 0, 10)
    α1 = pm.Normal('α1', 0, 10)
    μ = pm.Deterministic('μ', α0 + α1)
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=obs)
    m9_5 = sts.ulam()

print('m9.5:')
sts.precis(m9_5)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
