#!/usr/bin/env python3
# =============================================================================
#     File: 8H3.py
#  Created: 2023-09-21 11:12
#   Author: Bernie Roesler
#
"""
Chapter 8, Exercise 8H3.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

df = pd.read_csv(Path('../data/rugged.csv'))

# Remove NaNs
df = df.dropna(subset='rgdppc_2000')

# Normalize variables
df['rugged_std'] = df['rugged'] / df['rugged'].max()      # [0, 1]

df['log_GDP'] = np.log(df['rgdppc_2000'])
df['log_GDP_std'] = df['log_GDP'] / df['log_GDP'].mean()  # proportion of avg

df['cid'] = (df['cont_africa'] == 1).astype(int)

# Drop Seychelles from a version of the dataset
df_ns = df.loc[df['country'] != 'Seychelles'].copy()


# -----------------------------------------------------------------------------
#         8H3(a) Define the interaction model.
# -----------------------------------------------------------------------------
def build_model(data):
    """Fit the interaction model.

    .. math::
        y ~ N(μ, σ)
        μ ~ α + β_A A + β_R R + β_AR A R

    where A = `cont_africa`, R = `rugged`.
    """
    with pm.Model():
        R = pm.MutableData('R', data['rugged_std'])
        A = pm.MutableData('A', data['cont_africa'])  # index value
        obs = pm.MutableData('obs', data['log_GDP_std'])
        α = pm.Normal('α', 1, 0.1, shape=(2,))
        β = pm.Normal('β', 0, 0.3, shape=(2,))
        μ = pm.Deterministic('μ', α[A] + β[A]*R)  # de-mean R?
        σ = pm.Exponential('σ', 1)
        y = pm.Normal('y', μ, σ, observed=obs, shape=R.shape)
        return sts.quap(data=data)


# Fit with and without Seychelles
q_all = build_model(df)
q_ns = build_model(df_ns)

print(f"{float(q_all.coef['β'][1] / q_ns.coef['β'][1]) = :.2f}")  # ≈ 2.35

# We now have 4 options: (Non-)?African countries, and with(out)? Seychelles.
ct = sts.coef_table([q_all, q_ns], ['all', 'no Seychelles'])
sts.plot_coef_table(ct, fignum=1)


# -----------------------------------------------------------------------------
#         6H3(b) Plot posterior predictions
# -----------------------------------------------------------------------------
rugged_seq = np.linspace(-0.1, 1.1, 30)


def plot_linear_model(
    quap,
    is_Africa=True,
    has_Seychelles=True,
    annotate_lines=False,
    ax=None
):
    """Plot the data and associated model."""
    if ax is None:
        ax = plt.gca()

    df = quap.data

    if is_Africa:
        cid = np.ones_like(rugged_seq).astype(int)
        c = fc = 'C0'
        label = 'Africa'
    else:
        cid = np.zeros_like(rugged_seq).astype(int)
        c = 'k'
        fc = 'none'
        label = 'Not Africa'

    # NOTE explicitly evaluate the model, because lmplot expects the `x` kwarg
    # to be the same as the name of the independent variable in the model. In
    # this case, 'R' ≠ 'rugged_std', so we get an error.
    mu_samp = sts.lmeval(
        quap,
        out=quap.model.μ,
        eval_at={'R': rugged_seq, 'A': cid},
    )

    if not has_Seychelles:
        # Do not re-plot the data?
        c = fc = 'C3'
        sts.lmplot(
            fit_x=rugged_seq, fit_y=mu_samp,
            x='rugged_std', y='log_GDP_std',
            data=df.loc[df['cid'] == is_Africa],
            q=0.97,
            line_kws=dict(c=c),
            fill_kws=dict(facecolor=c),
            marker_kws=dict(edgecolor=c, facecolor='none', lw=2, s=50),
            label=label,
            ax=ax,
        )
    else:
        sts.lmplot(
            fit_x=rugged_seq, fit_y=mu_samp,
            x='rugged_std', y='log_GDP_std',
            data=df.loc[df['cid'] == is_Africa],
            q=0.97,
            line_kws=dict(c=c),
            fill_kws=dict(facecolor=c),
            marker_kws=dict(edgecolor=c, facecolor=fc, lw=2),
            label=label,
            ax=ax,
        )

    # Annotate lines
    if annotate_lines:
        ax.text(
            x=0.8,
            y=1.02*mu_samp.mean('draw')[int(0.8*len(mu_samp))],
            s=label,
            c=c
        )

    ax.spines[['right', 'top']].set_visible(False)

    return ax


# Plot the interaction
fig = plt.figure(3, clear=True, constrained_layout=True)
fig.set_size_inches((10, 5), forward=True)
fig.suptitle('Model with and without Seychelles')

gs = fig.add_gridspec(nrows=1, ncols=2)

sharey = None
for is_Africa in [False, True]:
    ax = fig.add_subplot(gs[~is_Africa], sharey=sharey)
    sharey = ax
    plot_linear_model(q_all, is_Africa, ax=ax)
    if is_Africa:
        plot_linear_model(q_ns, is_Africa, has_Seychelles=False, ax=ax)
        ax.legend(
            handles=ax.lines,
            labels=['with Seychelles', 'without Seychelles']
        )

    title = 'African Nations'
    if not is_Africa:
        title = 'Non-' + title
    ax.set(title=title,
           xlabel='ruggedness [std]',
           ylabel='log GDP (prop. of mean)')

    # Label countries manually since there does not seem to be a clear method
    # to the madness.
    if is_Africa:
        countries = [
            'Equatorial Guinea',
            'Seychelles',
            'South Africa',
            'Swaziland',
            'Lesotho',
            'Rwanda',
            'Burundi',
        ]
    else:
        countries = [
            'Luxembourg',
            'Switzerland',
            'Greece',
            'Lebanon',
            'Nepal',
            'Tajikistan',
            'Yemen',
        ]

    for c in countries:
        tf = df.loc[df['country'] == c]
        ax.text(
            x=float(tf['rugged_std'].iloc[0]) + 0.02,
            y=float(tf['log_GDP_std'].iloc[0]),
            s=c,
            ha='left',
            va='bottom',
        )


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
