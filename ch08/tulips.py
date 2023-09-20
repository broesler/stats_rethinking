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


# -----------------------------------------------------------------------------
#         Plot a triptych of shade values for the posterior (R code 8.25)
# -----------------------------------------------------------------------------
def plot_triptych(model, fig, N_lines=20):
    """Plot a triptych over model parameters."""
    gs = fig.add_gridspec(ncols=3)
    sharex = sharey = None
    vals = np.arange(-1, 2)

    for i, s in enumerate(vals):
        ax = fig.add_subplot(gs[i], sharex=sharex, sharey=sharey)
        sharex = sharey = ax

        ax.scatter(
            x='water_cent',
            y='blooms_std',
            data=df[df['shade_cent'] == s],
            c='C0',
            alpha=0.4
        )

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

    return ax


# Figure 8.7
fig = plt.figure(1, clear=True, constrained_layout=True)

models = [m8_6, m8_7]
mnames = ['m8.6 - no interaction', 'm8.7 - water + shade interaction']

fig.set_size_inches((10, 3*len(models)), forward=True)
subfigs = fig.subfigures(len(models), 1)

for subfig, model, mname in zip(subfigs, models, mnames):
    plot_triptych(model, subfig)
    subfig.suptitle(mname)

# =============================================================================
# =============================================================================
