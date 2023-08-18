#!/usr/bin/env python3
# =============================================================================
#     File: post_treatment_bias.py
#  Created: 2023-05-12 14:28
#   Author: Bernie Roesler
#
"""
§6.2 Post-Treatment Bias.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# -----------------------------------------------------------------------------
#         Simulate plant growth treatment (R code 6.14)
# -----------------------------------------------------------------------------
N = 100  # number of plants

# The "unknown" data-generating distribution
h0 = stats.norm(10, 2).rvs(N)  # initial heights
treatment = np.repeat([0, 1], N//2)
fungus = stats.bernoulli(0.5 - treatment * 0.4).rvs(N)
h1 = h0 + stats.norm(5 - 3 * fungus).rvs(N)

df = pd.DataFrame(np.c_[h0, h1, treatment, fungus],
                  columns=['h0', 'h1', 'treatment', 'fungus'])

sts.precis(df)

# -----------------------------------------------------------------------------
#         Create a Model
# -----------------------------------------------------------------------------
# The prior (R code 6.15)
sim_p = stats.lognorm(0.25).rvs(10_000)
sts.precis(pd.DataFrame(sim_p, columns=['sim_p']))

# The model (R code 6.16)
with pm.Model():
    α = pm.Lognormal('α', 0, 0.25)
    p = pm.Deterministic('p', α)
    μ = pm.Deterministic('μ', p * df['h0'])
    σ = pm.Exponential('σ', 1)
    h1 = pm.Normal('h1', μ, σ, observed=df['h1'])
    m6_6 = sts.quap(data=df)

print('m6.6:')
sts.precis(m6_6)

# Now model proportion of growth as a function of the predictors (R code 6.17)
with pm.Model():
    α = pm.Lognormal('α', 0, 0.2)
    βt = pm.Normal('βt', 0, 0.5)
    βf = pm.Normal('βf', 0, 0.5)
    p = pm.Deterministic('p', α + βt * df['treatment'] + βf * df['fungus'])
    μ = pm.Deterministic('μ', p * df['h0'])
    σ = pm.Exponential('σ', 1)
    h1 = pm.Normal('h1', μ, σ, observed=df['h1'])
    m6_7 = sts.quap(data=df)

print('m6.7:')
sts.precis(m6_7)

# ==> βt ~ 0! No effect? Wrong! We asked the question: Once we already know
# whether or not a plant developed fungus, does soil treatment matter? The
# answer is "no".

# Remove fungus from the model (R code 6.18)
with pm.Model():
    α = pm.Lognormal('α', 0, 0.2)
    βt = pm.Normal('βt', 0, 0.5)
    p = pm.Deterministic('p', α + βt * df['treatment'])
    μ = pm.Deterministic('μ', p * df['h0'])
    σ = pm.Exponential('σ', 1)
    h1 = pm.Normal('h1', μ, σ, observed=df['h1'])
    m6_8 = sts.quap(data=df)

print('m6.8:')
sts.precis(m6_8)

# Define DAG for causal effects (R code 6.19 - 6.21):
#
#  H0           T
#   \           ↓
#    \          F
#     \        /
#      -> H1 <-
#
# use algs.graph API?
#
# R> dseparated( plant_dag, "T", "H1")      == FALSE
# R> dseparated( plant_dag, "T", "H1", "F") == TRUE
#
# Asks "Is there a path from T to H1? That doesn't pass through F?"
#
# R> impliedConditionalIndependencies( plant_dag )
# F _||_ H0
# H0 _||_ T
# H1 _||_ T | F
#
# "There is no path from H0 to F"
# "There is no path from T to H0"
# "There is a path from T to H1, but it goes through F"

# ----------------------------------------------------------------------------- 
#         §7.5.1 Model Comparison
# -----------------------------------------------------------------------------
print(f"{sts.WAIC(m6_7)['h1'] = }")

models = [m6_6, m6_7, m6_8]
mnames = ['m6.6', 'm6.7', 'm6.8']

coeftab = sts.coef_table(models, mnames)
print(coeftab)
sts.plot_coef_table(coeftab, fignum=1)

cmp = sts.compare(models, mnames)
ct = cmp['ct']
with pd.option_context('display.precision', 2):
    print(ct)
sts.plot_compare(ct, fignum=2)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
