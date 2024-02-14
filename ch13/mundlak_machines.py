#!/usr/bin/env python3
# =============================================================================
#     File: mundlak_machines.py
#  Created: 2024-02-14 15:18
#   Author: Bernie Roesler
#
"""
See BONUS of Lecture 12 (2023) - Multilevel Models.

DAG:
    Xi -> Yi  # individuals (i.e. X = student prep, Y = test scores)
    G -> Xi   # unmeasured group traits (i.e. classroom noise)
    G -> Yi
    Z -> Yi   # measured group traits (i.e. classroom temperature)
    G -> Xg   # only in Mundlak Machine! Xg is mean of each group.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm

from scipy.special import expit

import stats_rethinking as sts

# Simulate data
N_groups = 30
N_id = 200
a0 = -2
bXY = 1.0
bZY = -0.5

rng = np.random.default_rng()

g = rng.choice(np.arange(N_groups), size=N_id)
Ug = rng.normal(1.5, size=N_groups)  # group confounds
Xv = rng.normal(Ug[g], size=N_id)     # individual varying trait
Zv = rng.normal(size=N_groups)        # group varying trait
Yv = rng.binomial(1, p=expit(a0 + bXY * Xv + Ug[g] + bZY*Zv[g]), size=N_id)

print(pd.DataFrame(dict(g=g)).value_counts().sort_index())

# Build a naïve model -- same intercept for *every* group
with pm.Model():
    a = pm.Normal('a', 0, 10)
    bxy = pm.Normal('bxy', 0, 1)
    bzy = pm.Normal('bzy', 0, 1)
    p = pm.math.invlogit(a + bxy * Xv + bzy * Zv[g])
    Y = pm.Bernoulli('Y', p, observed=Yv)
    m_naive = sts.ulam()

# Build a fixed effects model -- different intercepts for each group
#  * No pooling! Inefficient, but "soaks up" group-level confounding.
#    Y = expit( (a0 + Ug[g]) + X + bzy*Z[g] )
#    a[g] -> (a0 + Ug[g])
#  * Cannot identify any group-level effects (bzy distribution is very spread)
with pm.Model():
    a = pm.Normal('a', 0, 10, size=g.shape)
    bxy = pm.Normal('bxy', 0, 1)
    bzy = pm.Normal('bzy', 0, 1)
    p = pm.math.invlogit(a[g] + bxy * Xv + bzy * Zv[g])
    Y = pm.Bernoulli('Y', p, observed=Yv)
    m_fixed = sts.ulam()

# print('mf:')
# sts.precis(mf)

# Build a multilevel model:
#  * Partial pooling by learning the mean and variance parameters
#  * Better estimates for G
#  * Worse estimates for X
#  * Bonus: can identify Z!
with pm.Model():
    a_bar = pm.Normal('a_bar', 0, 1)
    τ = pm.Exponential('τ', 1)
    a = pm.Normal('a', a_bar, τ, size=g.shape)
    bxy = pm.Normal('bxy', 0, 1)
    bzy = pm.Normal('bzy', 0, 1)
    p = pm.math.invlogit(a[g] + bxy * Xv + bzy * Zv[g])
    Y = pm.Bernoulli('Y', p, observed=Yv)
    m_multi = sts.ulam()

# Build a Mundlak Machine:
#  * Partial pooling by learning the mean and variance parameters
#  * Include group average X
#  * Better X, but improper respect for uncertainty in Xbar

# Get average of each group
Xbar = pd.DataFrame(dict(g=g, X=Xv)).groupby('g').mean()['X']

with pm.Model():
    a_bar = pm.Normal('a_bar', 0, 1)
    τ = pm.Exponential('τ', 1)
    a = pm.Normal('a', a_bar, τ, size=g.shape)
    bxy = pm.Normal('bxy', 0, 1)
    bzy = pm.Normal('bzy', 0, 1)
    bxg = pm.Normal('bxg', 0, 1)
    p = pm.math.invlogit(a[g] + bxy * Xv + bzy * Zv[g] + bxg * Xbar[g])
    Y = pm.Bernoulli('Y', p, observed=Yv)
    m_mundk = sts.ulam()


# Latent Mundlak Machine -- estimate Xbar with observed Xis
#  * respects uncertainty in G
#  * a latent measurement error model for the group mean
with pm.Model():
    # Estimate Xi | do(G)
    # X is only influenced by the confound, so just a linear regression
    aX = pm.Normal('aX', 0, 1)
    bux = pm.Exponential('bux', 1)
    u = pm.Normal('u', 0, 1, size=g.shape)
    σ = pm.Exponential('σ', 1)
    μ = aX + bux * u[g]
    X = pm.Normal('X', μ, σ, observed=Xv)

    # Estimate Yi | do(X)
    a_bar = pm.Normal('a_bar', 0, 1)
    τ = pm.Exponential('τ', 1)
    a = pm.Normal('a', a_bar, τ, size=g.shape)
    bxy = pm.Normal('bxy', 0, 1)
    bzy = pm.Normal('bzy', 0, 1)
    buy = pm.Normal('buy', 0, 1)
    p = pm.math.invlogit(a[g] + bxy * X + bzy * Zv[g] + buy * u[g])
    Y = pm.Bernoulli('Y', p, observed=Yv)
    m_lamun = sts.ulam()


# -----------------------------------------------------------------------------
#         Plot distributions
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(num=1, ncols=2, clear=True)
fig.set_size_inches((10, 3.5), forward=True)

models = [m_naive, m_fixed, m_multi, m_mundk, m_lamun]
colors = ['gray', 'k', 'C3', 'C0', 'C2']
labels = ['naïve', 'fixed', 'multilevel', 'Mundlak', 'Latent Mundlak']

ax = axs[0]
ax.axvline(bXY, ls='--', c='gray', lw=1)

for m, c, label in zip(models, colors, labels):
    sns.kdeplot(m.samples['bxy'].values.flat, bw_adjust=0.5, c=c, ax=ax, label=label)

ax.set(xlabel='bXY',
       ylabel='Density')
ax.spines[['top', 'right']].set_visible(False)

ax = axs[1]
ax.axvline(bZY, ls='--', c='gray', lw=1)

for m, c, label in zip(models, colors, labels):
    sns.kdeplot(m.samples['bzy'].values.flat, bw_adjust=0.5, c=c, ax=ax, label=label)

ax.legend()
ax.set(xlabel='bZY',
       ylabel='Density')
ax.spines[['top', 'right']].set_visible(False)


# =============================================================================
# =============================================================================
