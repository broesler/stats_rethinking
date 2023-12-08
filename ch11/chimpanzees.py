#!/usr/bin/env python3
# =============================================================================
#     File: chimpanzees.py
#  Created: 2023-12-05 10:51
#   Author: Bernie Roesler
#
"""
§11.1 Chimpanzees with social choice. Logistic Regression.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats
from scipy.special import logit, expit

import stats_rethinking as sts

# Get the data (R code 11.1), forcing certain dtypes
df = pd.read_csv(
    Path('../data/chimpanzees.csv'),
    dtype=dict({
        'actor': int,
        'condition': bool,
        'prosoc_left': bool,
        'chose_prosoc': bool,
        'pulled_left': bool,
    })
)

df['actor'] -= 1  # python is 0-indexed

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 504 entries, 0 to 503
# Data columns (total 8 columns):
#    Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   actor         504 non-null    category
#  1   recipient     252 non-null    float64
#  2   condition     504 non-null    bool
#  3   block         504 non-null    int64
#  4   trial         504 non-null    int64
#  5   prosoc_left   504 non-null    bool
#  6   chose_prosoc  504 non-null    bool
#  7   pulled_left   504 non-null    bool
# dtypes: bool(4), category(1), float64(1), int64(2)
# memory usage: 14.5 KB

# Define a treatment index variable combining others (R code 11.2)
df['treatment'] = df['prosoc_left'] + 2 * df['condition']  # in range(4)

# (R code 11.3)
# print(df.pivot_table(
#     index='treatment',
#     values=['prosoc_left', 'condition'],
#     aggfunc='sum',
# ))

# Build a simple model (R code 11.4)
with pm.Model():
    a = pm.Normal('a', 0, 10)
    p = pm.Deterministic('p', pm.math.invlogit(a))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_1 = sts.quap(data=df)

bad_prior = m11_1.sample_prior(N=10_000).sortby('p')
bad_dens = stats.gaussian_kde(bad_prior['p'], bw_method=0.01).pdf(bad_prior['p'])

# Model with better prior
with pm.Model():
    a = pm.Normal('a', 0, 1.5)
    p = pm.Deterministic('p', pm.math.invlogit(a))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_1 = sts.quap(data=df)

reg_prior = m11_1.sample_prior(N=10_000).sortby('p')
reg_dens = stats.gaussian_kde(reg_prior['p'], bw_method=0.01).pdf(reg_prior['p'])

# -----------------------------------------------------------------------------
#         FIgure 11.3 (R code 11.6)
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(num=1, ncols=2, clear=True)
fig.set_size_inches((10, 5), forward=True)

ax = axs[0]

ax.plot(bad_prior['p'], bad_dens, c='k', label=r"$a \sim \mathcal{N}(0, 10)$")
ax.plot(reg_prior['p'], reg_dens, c='C0', label=r"$a \sim \mathcal{N}(0, 1.5)$")

ax.legend()
ax.set(xlabel="prior probability 'pulled_left'",
       ylabel='Density')
ax.spines[['top', 'right']].set_visible(False)


# -----------------------------------------------------------------------------
#         Include b effect (R code 11.7)
# -----------------------------------------------------------------------------
with pm.Model():
    a = pm.Normal('a', 0, 1.5)
    b = pm.Normal('b', 0, 10, shape=(4,))
    p = pm.Deterministic('p', pm.math.invlogit(a + b[df['treatment']]))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_2 = sts.quap(data=df)

# TODO not sure why sampling the prior gives a `p_dim_0` = 508.
# Write into pymc help as to why we need to explicitly compute the function
# that has already been defined in the model.

# Get the difference in treatments (R code 11.8)
bad_prior = m11_2.sample_prior(N=10_000)
p = expit(bad_prior['a'] + bad_prior['b'])
bad_diff = np.abs(p[:, 0] - p[:, 1]).sortby(lambda x: x)
bad_dens = stats.gaussian_kde(bad_diff, bw_method=0.01).pdf(bad_diff)

# More regularizing prior on b (R code 11.9)
with pm.Model():
    a = pm.Normal('a', 0, 1.5)
    b = pm.Normal('b', 0, 0.5, shape=(4,))
    p = pm.Deterministic('p', pm.math.invlogit(a + b[df['treatment']]))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_3 = sts.quap(data=df)

reg_prior = m11_3.sample_prior(N=10_000)
p = expit(reg_prior['a'] + reg_prior['b'])
reg_diff = np.abs(p[:, 0] - p[:, 1]).sortby(lambda x: x)
reg_dens = stats.gaussian_kde(reg_diff, bw_method=0.01).pdf(reg_diff)

# Plot on the right side
ax = axs[1]
ax.plot(bad_diff, bad_dens, c='k', label=r"$b \sim \mathcal{N}(0, 10)$")
ax.plot(reg_diff, reg_dens, c='C0', label=r"$b \sim \mathcal{N}(0, 1.5)$")

ax.legend()
ax.set(xlabel="prior probability of\ndifference between treatments",
       ylabel='Density')
ax.spines[['top', 'right']].set_visible(False)

# -----------------------------------------------------------------------------
#         Create the model with actor now (R Code 11.10)
# -----------------------------------------------------------------------------
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    a = pm.Normal('a', 0, 1.5, shape=(len(df['actor'].unique()),))      # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    p = pm.Deterministic('p', pm.math.invlogit(a[actor] + b[treatment]))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_4 = sts.ulam(data=df)

print('m11.4:')
sts.precis(m11_4)

# (R code 11.11)
post = m11_4.get_samples()
p_left = expit(post['a'])
p_left.name = 'p'

fig, ax = sts.plot_precis(p_left, mname='m11_4', fignum=2)

fig.set_size_inches((6, 3), forward=True)
ax.axvline(0.5, ls='--', c='k')

# Plot the treatment effects (R code 11.12)
labels = ['R/N', 'L/N', 'R/P', 'L/P']
fig, _ = sts.plot_precis(post['b'], fignum=3, labels=labels)
fig.set_size_inches((6, 3), forward=True)

# Plot the contrasts in the treatments (R code 11.13)
post_b = (
    post['b']
    .assign_coords(b_dim_0=labels)
    .stack(sample=('chain', 'draw'))
    .transpose('sample', ...)
)

diffs = dict(
    dbR=post_b.sel(b_dim_0='R/N') - post_b.sel(b_dim_0='R/P'),
    dbL=post_b.sel(b_dim_0='L/N') - post_b.sel(b_dim_0='L/P')
)

fig, ax = sts.plot_precis(pd.DataFrame(diffs), fignum=4)
fig.set_size_inches((12, 6), forward=True)
ax.set_ylim((-0.5, 1.5))  # give more space around lines


# -----------------------------------------------------------------------------
#         (R code 11.14) Posterior preditictive check of pulled_left proportions
# -----------------------------------------------------------------------------

# TODO transpose the entire plot? "left/right" makes more sense that way.
def plot_actors(pf, ci=None, title='', c='C0', ax=None):
    """Plot the output proportions for each actor and treatment."""
    if ax is None:
        ax = plt.gca()

    # Dividing line of left/right preference
    ax.axhline(0.5, ls='--', lw=1, color='k', alpha=0.5)

    # Label the treatment indices for easier use
    pf = pf.copy()
    pf.index = pf.index.set_levels(labels, level='treatment')

    if ci is not None:
        ci = ci.copy()
        ci = ci.transpose(..., 'quantile').to_pandas()
        ci.index = pf.index
        errs = ci.sub(pf, axis='rows').abs().T  # (2, N) for errorbar

    # Plot each actor as a "column"
    N_actors = pf.index.get_level_values('actor').unique().size
    N = len(pf)
    for i in range(N_actors):
        # Plot divider
        ax.axvline(4*i + 4.5, c='k', lw=1)
        # Plot "title"
        ax.text(4*i + 2.5, 1.05, f"actor {i+1}", ha='center', va='bottom')

        # Plot left/right data offset from each other for clarity
        if i != 1:
            ax.plot(4*i + np.r_[1, 3], pf.loc[i, ['R/N', 'R/P']], ls='--', c=c)
            ax.plot(4*i + np.r_[2, 4], pf.loc[i, ['L/N', 'L/P']], c=c)

    # Plot all the points at once
    xs = np.arange(1, N, 4)
    # ax.scatter(xs,   pf.loc[:, 'R/N'], ec=c, fc='white')
    # ax.scatter(xs+1, pf.loc[:, 'L/N'], ec=c, fc='white')
    # ax.scatter(xs+2, pf.loc[:, 'R/P'], fc=c)
    # ax.scatter(xs+3, pf.loc[:, 'L/P'], fc=c)

    yoff = 0.05
    # ax.text(1, pf.loc[0, 'R/N'] - yoff, 'R/N', ha='center', va='top')
    # ax.text(2, pf.loc[0, 'L/N'] + yoff, 'L/N', ha='center', va='bottom')
    # ax.text(3, pf.loc[0, 'R/P'] - yoff, 'R/P', ha='center', va='top')
    # ax.text(4, pf.loc[0, 'L/P'] + yoff, 'L/P', ha='center', va='bottom')

    for i, s in enumerate(labels):
        if ci is None:
            # Plot the data points, closed for 'participant', open otherwise
            ax.scatter(xs + i, pf.loc[:, s],
                       fc=(c if 'P' in s else 'white'), ec=c, zorder=2)
            # Annotate the first four points
            sign, va = (-1, 'top') if 'R' in s else (1, 'bottom')
            ax.text(x=i + 1, y=pf.loc[0, s] + sign*yoff,
                    s=s, ha='center', va=va, c=c)
        else:
            # PLot the data, but with errorbars now
            ax.errorbar(xs + i, pf.loc[:, s],
                        yerr=errs.loc[:, pd.IndexSlice[:, s]],
                        fmt='o', c=c, mfc=(c if 'P' in s else 'white'), mec=c)

    ax.set(
        ylabel='proportion left lever',
        xlim=(0, 29),
        ylim=(-0.05, 1.05),
        yticks=[0, 0.5, 1],
    )
    ax.set_title(title, y=1.15)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.spines[['top', 'right']].set_visible(False)

    return ax


# Plot these values vs the posterior predictions
fig, axs = plt.subplots(num=4, nrows=2, clear=True)

# Plot the observed proportions
pf = df.groupby(['actor', 'treatment']).mean()['pulled_left']

plot_actors(pf, title='observed proportions', ax=axs[0])

# Get the posterior approximations at each point
p_samp = sts.lmeval(
    m11_4,
    out=m11_4.model.p,
    dist=post,
    eval_at=dict({x: pf.index.get_level_values(x).values
                  for x in ['actor', 'treatment']}),
)

p_mean = p_samp.mean('draw').to_series()
p_mean.index = pf.index

q = 0.89
a = (1 - q)/2
p_ci = p_samp.quantile([a, 1-a], dim='draw')

plot_actors(p_mean, ci=p_ci, title='posterior predictions',  c='k', ax=axs[1])


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
