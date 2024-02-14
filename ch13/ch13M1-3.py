#!/usr/bin/env python3
# =============================================================================
#     File: ch13M1-3.py
#  Created: 2024-02-14 16:30
#   Author: Bernie Roesler
#
"""
Ch. 13 Exercises M1-2 (mislabelled in the book as 12M1-3).
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

df = pd.read_csv(
    Path('../data/reedfrogs.csv'),
    dtype=dict(pred='category', size='category')
)

# Approximate the tadpole mortality in each tank (R code 13.2)
df['tank'] = df.index

# -----------------------------------------------------------------------------
#         13M1. Model main effects and interaction
# -----------------------------------------------------------------------------
# Predator predictor only
# See Lecture 12 (2023) @ 43:52
with pm.Model():
    N = pm.ConstantData('N', df['density'])
    P = pm.ConstantData('P', df['pred'].cat.codes)
    tank = pm.ConstantData('tank', df['tank'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    β_P = pm.Normal('β_P', 0, 0.5)
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', α_bar, σ, shape=tank.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[tank] + β_P * P))
    S = pm.Binomial('S', N, p, observed=df['surv'])
    mP = sts.ulam(data=df)

# Size predictor
with pm.Model():
    N = pm.ConstantData('N', df['density'])
    Z = pm.ConstantData('Z', df['size'].cat.codes)
    tank = pm.ConstantData('tank', df['tank'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    β_Z = pm.Normal('β_Z', 0, 0.5)
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', α_bar, σ, shape=tank.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[tank] + β_Z * Z))
    S = pm.Binomial('S', N, p, observed=df['surv'])
    mZ = sts.ulam(data=df)

# Predator + Size predictor
with pm.Model():
    N = pm.ConstantData('N', df['density'])
    P = pm.ConstantData('P', df['pred'].cat.codes)
    Z = pm.ConstantData('Z', df['size'].cat.codes)
    tank = pm.ConstantData('tank', df['tank'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    β_P = pm.Normal('β_P', 0, 0.5)
    β_Z = pm.Normal('β_Z', 0, 0.5)
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', α_bar, σ, shape=tank.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[tank] + β_P*P + β_Z*Z))
    S = pm.Binomial('S', N, p, observed=df['surv'])
    mPZ = sts.ulam(data=df)

# Predator + Size + interaction
with pm.Model():
    N = pm.ConstantData('N', df['density'])
    P = pm.ConstantData('P', df['pred'].cat.codes)
    Z = pm.ConstantData('Z', df['size'].cat.codes)
    tank = pm.ConstantData('tank', df['tank'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    β_P = pm.Normal('β_P', 0, 0.5)
    β_Z = pm.Normal('β_Z', 0, 0.5)
    β_PZ = pm.Normal('β_PZ', 0, 0.5)
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', α_bar, σ, shape=tank.shape)
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(α[tank] + β_P*P + β_Z*Z + β_PZ*P*Z)
    )
    S = pm.Binomial('S', N, p, observed=df['surv'])
    mPZI = sts.ulam(data=df)


# -----------------------------------------------------------------------------
#         Plot model results
# -----------------------------------------------------------------------------
def plot_model(model, cat='pred', cat_names=None, colors=None, labels=None,
               ax=None):
    if ax is None:
        ax = plt.gca()

    if cat_names is None:
        cat_names = ['no', 'pred']

    if labels is None:
        labels = cat_names

    if colors is None:
        colors = ['C0', 'C3']

    p_est = model.deterministics['p'].mean(('chain', 'draw'))
    a = (1 - 0.89) / 2
    p_PI = model.deterministics['p'].quantile([a, 1-a], dim=('chain', 'draw'))

    # Label tank sizes
    ax.axvline(15.5, ls='--', c='gray', lw=1)
    ax.axvline(31.5, ls='--', c='gray', lw=1)
    ax.text(     8, 0, 'small tanks (10)',  ha='center', va='bottom')
    ax.text(15 + 8, 0, 'medium tanks (25)', ha='center', va='bottom')
    ax.text(31 + 8, 0, 'large tanks (35)',  ha='center', va='bottom')

    ax.set(xlabel='tank',
           ylabel='proportion survival',
           ylim=(-0.05, 1.05))
    ax.spines[['top', 'right']].set_visible(False)

    # Plot the data
    ax.scatter('tank', 'propsurv', data=df, c='k', label='data')

    for p, c in zip(cat_names, colors):
        tf = df.loc[df[cat] == p]
        ax.errorbar(
            tf['tank'],
            p_est.sel(p_dim_0=tf.index),
            yerr=np.abs(p_PI - p_est).sel(p_dim_0=tf.index),
            marker='o', ls='none', c=c, lw=3, alpha=0.7,
            label=labels[0] if p == cat_names[0] else labels[1],
        )

    ax.legend()


fig, ax = plt.subplots(num=1, clear=True)
plot_model(mP, ax=ax)

fig, ax = plt.subplots(num=2, clear=True)
plot_model(mZ, cat='size', cat_names=df['size'].cat.categories,
           colors=['C1', 'C2'], ax=ax)


# -----------------------------------------------------------------------------
#         13M2. Compare the models
# -----------------------------------------------------------------------------
models = [mP, mZ, mPZ, mPZI]
mnames = ['P', 'Z', 'P + Z', 'P + Z + P*Z']
ct = sts.compare(models, mnames, ic='PSIS', sort=True)
print(ct['ct'])


# -----------------------------------------------------------------------------
#         Plot the distributions of σ
# -----------------------------------------------------------------------------
# σ is the hyperparameter that describes the inferred variation across tanks.
fig, ax = plt.subplots(num=3, clear=True)
for m, label in zip(models, mnames):
    sns.kdeplot(m.samples['σ'].values.flat, bw_adjust=0.5, ax=ax, label=label)
ax.legend()
ax.set(xlabel='σ',
       ylabel='Density')
ax.spines[['top', 'right']].set_visible(False)

# NOTE the variation among tanks is largely explained away by the predator
# variable P, but not by the size variable Z.

# =============================================================================
# =============================================================================
