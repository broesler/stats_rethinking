#!/usr/bin/env python3
# =============================================================================
#     File: tulips.py
#  Created: 2023-09-19 15:32
#   Author: Bernie Roesler
#
"""
§8.3 Continuous interactions.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

df = pd.read_csv(Path('../data/tulips.csv'))
df['bed'] = df['bed'].astype('category')

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 27 entries, 0 to 26
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   bed     27 non-null     category
#  1   water   27 non-null     int64
#  2   shade   27 non-null     int64
#  3   blooms  27 non-null     float64
# dtypes: category(1), float64(1), int64(2)
# memory usage: 939.0 bytes

# Normalize variables (R code 8.20)
df['blooms_std'] = df['blooms'] / df['blooms'].max()
df['water_cent'] = df['water'] - df['water'].mean()
df['shade_cent'] = df['shade'] - df['shade'].mean()

# Check prior bounds on α
Ns = 10_000
a = stats.norm(0.5, 1).rvs(Ns)
print(f"P(N(0.5, 1) outside range) = {np.sum((a < 0) | (a > 1)) / Ns:.4f}")

a = stats.norm(0.5, 0.25).rvs(Ns)
print(f"P(N(0.5, 0.25) outside range) = {np.sum((a < 0) | (a > 1)) / Ns:.4f}")

# Build the model (R code 8.23)
with pm.Model():
    water = pm.MutableData('water', df['water_cent'])
    shade = pm.MutableData('shade', df['shade_cent'])
    α = pm.Normal('α', 0.5, 0.25)
    βw = pm.Normal('βw', 0, 0.25)
    βs = pm.Normal('βs', 0, 0.25)
    μ = pm.Deterministic('μ', α + βw*water + βs*shade)
    σ = pm.Exponential('σ', 1)
    blooms_std = pm.Normal('blooms_std', μ, σ, observed=df['blooms_std'])
    m8_6 = sts.quap(data=df)

print('m8.6:')
sts.precis(m8_6)

# Add an interaction term between shade and water (R code 8.24)
with pm.Model():
    water = pm.MutableData('water', df['water_cent'])
    shade = pm.MutableData('shade', df['shade_cent'])
    α = pm.Normal('α', 0.5, 0.25)
    βw = pm.Normal('βw', 0, 0.25)
    βs = pm.Normal('βs', 0, 0.25)
    βws = pm.Normal('βws', 0, 0.25)
    μ = pm.Deterministic('μ', α + βw*water + βs*shade + βws*water*shade)
    σ = pm.Exponential('σ', 1)
    blooms_std = pm.Normal('blooms_std', μ, σ, observed=df['blooms_std'])
    m8_7 = sts.quap(data=df)

print('m8.7:')
sts.precis(m8_7)

# Plot a triptych of shade values (R code 8.25)
N_lines = 20

# TODO refactor to plot m8.7 in the same figure
fig = plt.figure(1, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=3)

model = m8_6
sharex = sharey = None
vals = np.arange(-1, 2)
for i, s in enumerate(range(-1, 2)):
    ax = fig.add_subplot(gs[i], sharex=sharex, sharey=sharey)
    sharex = sharey = ax
    ax.scatter('water_cent', 'blooms_std', data=df[df['shade_cent'] == s],
               c='C0', alpha=0.4)
    mu_samp = sts.lmeval(
        model,
        out=model.model.μ,
        eval_at={'shade': s * np.ones(3), 'water': vals},
        N=N_lines,
    )
    ax.plot(vals, mu_samp, c='k', alpha=0.3)
    ax.set(title=f"shade = {s}",
           xlabel='water')
    ax.set_xticks(vals)
    ax.set_yticks([0, 0.5, 1])
    if i == 0:
        ax.set_ylabel('blooms')
    else:
        ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('m8.6')


# =============================================================================
# =============================================================================
