#!/usr/bin/env python3
# =============================================================================
#     File: ocean_tools_2.py
#  Created: 2024-01-17 11:29
#   Author: Bernie Roesler
#
"""
§12.1.2 Gamma-Poisson (negative-binomial) models.

See also §11.2.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

# Load the data
df = pd.read_csv(Path('../data/Kline.csv'))

df['contact_id'] = (df['contact'] == 'high').astype(int)

# Repeat the pure Poisson model for comparison
# (R code 11.52)
with pm.Model() as model:
    P = pm.MutableData('P', df['population'])
    cid = pm.MutableData('cid', df['contact_id'])
    α = pm.Normal('α', 1, 1, shape=(2,))
    β = pm.Exponential('β', 1, shape=(2,))
    γ = pm.Exponential('γ', 1)
    λ = pm.Deterministic('λ', pm.math.exp(α[cid]) * P**β[cid] / γ)
    T = pm.Poisson('T', λ, observed=df['total_tools'])
    m11_11 = sts.ulam(data=df)

# -----------------------------------------------------------------------------
#         Build a gamma-Poisson model (R code 12.6)
# -----------------------------------------------------------------------------
with pm.Model() as model:
    P = pm.MutableData('P', df['population'])
    cid = pm.MutableData('cid', df['contact_id'])
    α = pm.Normal('α', 1, 1, shape=(2,))
    β = pm.Exponential('β', 1, shape=(2,))
    γ = pm.Exponential('γ', 1)
    λ = pm.Deterministic('λ', pm.math.exp(α[cid]) * P**β[cid] / γ)
    # Add additional parameter for Gamma-Poisson distribution
    φ = pm.Exponential('φ', 1)
    T = pm.NegativeBinomial('T', mu=λ, alpha=φ, observed=df['total_tools'])
    m12_3 = sts.ulam(data=df)


# -----------------------------------------------------------------------------
#         Plot the Results
# -----------------------------------------------------------------------------
# Un-standardized values on real scale
Ns = 1000
Pseq = np.linspace(-5, 3, Ns)  # standardized scale
pop_seq = np.exp(
    sts.unstandardize(
        Pseq,
        np.log(df['population'])
    )
).astype(int)


def plot_data(ax, model, x, xv, topK=0):
    """Plot the data and mean fit, sized by the Pareto k value.

    Parameters
    ----------
    ax : Axes
        The axes in which to make the plot.
    model : PostModel
        The model to evaluate.
    x : str
        The name of the column of x-data.
    xv : array_like
        The x-values over which the mean is evaluated.
    topK : int >= 0, optional
        The number of Pareto k values to label.

    Returns
    -------
    None
    """
    # Get the scaling with the original data
    def reset_data():
        pm.set_data(
            dict(P=df['population'],
                cid=df['contact_id']),
            model=model.model
        )

    reset_data()
    loo = sts.LOOIS(model, pointwise=True)['T']
    pk = loo['pareto_k'] / loo['pareto_k'].max()

    # Closed circles for high contact, open circles for low contact
    ax.scatter(x, 'total_tools', data=df,
               ec='C0', fc=np.where(df['contact'] == 'high', 'C0', 'none'),
               s=1 + 10*np.exp(3*pk))

    for c, ls in zip([0, 1], ['--', '-']):
        λ_samp = sts.lmeval(
            model,
            out=model.model.λ,
            eval_at=dict(P=pop_seq, cid=np.full_like(pop_seq, c, dtype=int)),
        )
        sts.lmplot(
            fit_x=xv, fit_y=λ_samp,
            ax=ax,
            label=('high' if c else 'low') + ' contact',
            line_kws=dict(c='k', ls=ls),
            fill_kws=dict(fc='k', alpha=0.1),
        )

    # Label top K
    if topK:
        idx = loo.sort_values('pareto_k', ascending=False).index[:topK]
        for i in idx:
            name, tx, ty = df.loc[i, ['culture', x, 'total_tools']]
            k = loo.loc[i, 'pareto_k']
            ax.text(s=f"{name} ({k:.2f})", x=tx, y=ty+2,
                    ha='center', va='bottom')

    ax.spines[['top', 'right']].set_visible(False)
    reset_data()
    return


fig, axs = plt.subplots(num=1, ncols=2, sharex=True, sharey=True, clear=True)

ax = axs[0]
plot_data(ax, model=m11_11, x='population', xv=pop_seq)

ax.legend(loc='lower right')
ax.set(
    title='Poisson Model 11.11',
    xlabel='population',
    ylabel='total tools',
    xticks=[0, 50_000, 150_000, 250_000],
    xlim=(-10_000, 300_000),
    ylim=(0, 90),
    # yscale='log',
    # xscale='log',
)
ax.xaxis.set_major_formatter('{x:,d}')

ax = axs[1]
plot_data(ax, model=m12_3, x='population', xv=pop_seq)

ax.set(
    title='Gamma-Poisson Model 12.3',
    xlabel='population',
    xticks=[0, 50_000, 150_000, 250_000],
    xlim=(-10_000, 300_000),
    # xscale='log',
)
ax.xaxis.set_major_formatter('{x:,d}')

# =============================================================================
# =============================================================================
