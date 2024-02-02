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
import xarray as xr

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


def plot_linear_model(quap, is_Africa=True, has_Seychelles=True, ax=None):
    """Plot the data and associated model."""
    if ax is None:
        ax = plt.gca()

    df = quap.data

    if is_Africa:
        cid = np.ones_like(rugged_seq).astype(int)
        if has_Seychelles:
            label = 'Africa'
            c = fc = 'C0'
        else:
            label = 'Africa without Seychelles'
            c = 'C3'
            fc = 'none'
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

    sts.lmplot(
        fit_x=rugged_seq, fit_y=mu_samp,
        x='rugged_std', y='log_GDP_std', data=df.loc[df['cid'] == is_Africa],
        q=0.97,
        line_kws=dict(c=c),
        fill_kws=dict(facecolor=c),
        marker_kws=dict(edgecolor=c, facecolor=fc, lw=2),
        label=label,
        ax=ax,
    )

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

    if has_Seychelles:
        for c in countries:
            tf = df.loc[df['country'] == c]
            ax.text(
                x=float(tf['rugged_std'].iloc[0]) + 0.02,
                y=float(tf['log_GDP_std'].iloc[0]),
                s=c,
                ha='left',
                va='bottom',
            )

    ax.spines[['right', 'top']].set_visible(False)

    return ax


# Plot the interaction
fig = plt.figure(2, clear=True, constrained_layout=True)
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


# ----------------------------------------------------------------------------- 
#         8H3(c) Model comparison without Seychelles
# -----------------------------------------------------------------------------
# TODO justify priors?
def seychelles_model(opt='rugged'):
    """Build a model of GDP vs ruggedness without Seychelles.

    Parameters
    ----------
    opt : str in 'rugged', 'both', 'interaction'
        Type of model to build.
        'rugged' : μ = α + β_R R
        'both' : μ = α + β_A A + β_R R
        'interaction' : μ = α + β_A A + β_R R + β_AR A R.

    Returns
    -------
    quap : :obj:`Quap`
        The quadratic approximation to the posterior.
    """
    data = df_ns
    with pm.Model():
        R = pm.MutableData('R', data['rugged_std'])
        obs = pm.MutableData('obs', data['log_GDP_std'])
        α = pm.Normal('α', 1, 0.1)
        β_R = pm.Normal('β_R', 0, 0.3)
        if opt in ['both', 'interaction']:
            A = pm.MutableData('A', data['cont_africa'])
            β_A = pm.Normal('β_A', 0, 0.3)
        match opt:
            case 'rugged':
                μ = pm.Deterministic('μ', α + β_R*R)
            case 'both':
                μ = pm.Deterministic('μ', α + β_A*A + β_R*R)
            case 'interaction':
                β_AR = pm.Normal('β_AR', 0, 0.3)
                μ = pm.Deterministic('μ', α + β_A*A + β_R*R + β_AR*A*R)
        σ = pm.Exponential('σ', 1)
        y = pm.Normal('y', μ, σ, observed=obs, shape=R.shape)
        return sts.quap(data=data)


mnames = ['rugged', 'both', 'interaction']
models = [seychelles_model(x) for x in mnames]

cmp = sts.compare(models, mnames)
sts.plot_compare(cmp['ct'], fignum=3)

# Plot the model-averaged predictions and compare to just the interaction.
mean_samples = xr.Dataset()
for quap, name in zip(models, mnames):
    eval_at = {'R': rugged_seq}
    if name in ['both', 'interaction']:
        eval_at.update({'A': np.ones_like(rugged_seq).astype(int)})
    mean_samples[name] = sts.lmeval(quap, out=quap.model.μ, eval_at=eval_at)

weighted_μ = (
    (mean_samples * cmp['ct']['weight']['y'])
    .to_array(dim='model')
    .sum('model')  # weights already normalized, so just sum
)

fig, ax = plt.subplots(num=4, clear=True, constrained_layout=True)

# Plot model from part (b)
plot_linear_model(q_ns, is_Africa=True, has_Seychelles=False, ax=ax)

# Plot the weighted model predictions
c = fc = 'C0'
sts.lmplot(
    fit_x=rugged_seq, fit_y=weighted_μ,
    q=0.97,
    line_kws=dict(c=c),
    fill_kws=dict(facecolor=c),
    marker_kws=dict(edgecolor=c, facecolor=fc, lw=2),
    label='Model-Averaged',
    ax=ax,
)

ax.set(title='Model-averaged Predictions',
       xlabel='ruggedness [std]',
       ylabel='log GDP (proportion of mean)')
ax.legend()

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
