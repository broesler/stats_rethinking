#!/usr/bin/env python3
# =============================================================================
#     File: grad_school_2.py
#  Created: 2024-01-16 19:45
#   Author: Bernie Roesler
#
"""
§12.1 Over-Dispersed models.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from cycler import cycler
from pathlib import Path
from scipy import stats
from scipy.special import expit

import stats_rethinking as sts


# -----------------------------------------------------------------------------
#         Test plot of beta distribution
# -----------------------------------------------------------------------------
def beta2(pbar, theta):
    # Convert to standard parameterization
    a = pbar * θ
    b = (1 - pbar) * θ
    return stats.beta(a, b)


x = np.linspace(0, 1, 200)
pbars = np.arange(0.1, 1, 0.1)
θs = [0.1, 1, 2, 3, 5, 10, 30, 50, 100]

fig, axs = plt.subplots(num=1, ncols=2, sharex=True, sharey=True, clear=True)
fig.set_size_inches((10, 5), forward=True)

ax = axs[0]
ax.set_prop_cycle(
    cycler('color', plt.cm.viridis(np.linspace(0, 0.9, len(pbars))))
)

# Vary pbar
θ = 10
for pbar in pbars:
    ax.plot(x, beta2(pbar, θ).pdf(x),
            label=(f"{pbar = :g}, "
                   rf"($\alpha$ = {pbar*θ:.1f}, $\beta$ = {(1 - pbar)*θ:.1f})")
            )

ax.legend()
ax.set(xlabel='x',
       ylabel='density',
       xlim=(0, 1))
ax.spines[['top', 'right']].set_visible(False)

# Vary θ
ax = axs[1]
ax.set_prop_cycle(
    cycler('color', plt.cm.viridis(np.linspace(0, 0.9, len(θs))))
)

pbar = 0.3
ax.axvline(pbar, c='k', ls='--', lw=1, alpha=0.3)

for θ in θs:
    ax.plot(x, beta2(pbar, θ).pdf(x),
            label=(f"{θ = :3g}, "
                   rf"($\alpha$ = {pbar*θ:.1f}, $\beta$ = {(1 - pbar)*θ:.1f})")
            )

ax.legend()
ax.set(xlabel='x')
ax.spines[['top', 'right']].set_visible(False)


# ----------------------------------------------------------------------------- 
#         Model the UCB Admissions data
# -----------------------------------------------------------------------------
df = (pd.read_csv(Path('../data/UCBadmit.csv'))
    .reset_index()
    .rename({'index': 'case'}, axis='columns')
)

# Reorganize
df = (df
    .assign(
        dept=df['dept'].astype('category'),
        gender=df['applicant.gender'].astype('category')
    )
    .drop('applicant.gender', axis='columns')
)

# Create integer index column
df['gid'] = df['gender'].cat.codes

# Define the model
with pm.Model() as model:
    N = df['applications']
    a = pm.Normal('a', 0, 1.5, shape=(2,))
    pbar = pm.Deterministic('pbar', pm.math.invlogit(a[df['gid']]))
    θ = pm.Exponential('θ', 1)
    α, β = pbar*θ, (1-pbar)*θ
    admit = pm.BetaBinomial('admit', n=N, alpha=α, beta=β, observed=df['admit'])
    m12_1 = sts.ulam(data=df)

post = m12_1.get_samples()
post['da'] = post['a'].diff('a_dim_0').squeeze()
sts.precis(post)

# ----------------------------------------------------------------------------- 
#         Plot posterior distributions
# -----------------------------------------------------------------------------
gid = 0  # female applicants
# sample_dims = ('chain', 'draw')

pbar = expit(post['a'].sel(a_dim_0=gid)).mean()
θ = post['θ'].mean()

fig = plt.figure(2, clear=True)
ax = fig.add_subplot()
ax.plot(x, beta2(pbar, θ).pdf(x), 'k-', lw=2, label='mean distribution')

# Plot 50 draws from the posterior
for i in range(50):
    pbar = expit(post['a'].sel(chain=0, draw=i, a_dim_0=gid))
    θ = post['θ'].sel(chain=0, draw=i)
    ax.plot(x, beta2(pbar, θ).pdf(x), 'k-', lw=1, alpha=0.2)

ax.set(title='distribution of female admission rates',
       xlabel='probability of admission',
       ylabel='density',
       ylim=(0, 3))
ax.spines[['top', 'right']].set_visible(False)

# Plot the predictions
ax = sts.postcheck(
    m12_1,
    mean_name='pbar',
    agg_name='applications',
    major_group='dept',
    minor_group='gender',
    fignum=3,
)
ax.set_title('Model Beta-Binomial')
ax.set_ylim((0, 1))
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# =============================================================================
# =============================================================================
