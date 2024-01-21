#!/usr/bin/env python3
# =============================================================================
#     File: trolley.py
#  Created: 2024-01-18 17:57
#   Author: Bernie Roesler
#
"""
§12.3 Ordered Categorical Outcomes: The Trolley Problem.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy.special import logit, expit

import stats_rethinking as sts

df = pd.read_csv(Path('../data/Trolley.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 9930 entries, 0 to 9929
# Data columns (total 12 columns):
#    Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   case       9930 non-null   object
#  1   response   9930 non-null   int64
#  2   order      9930 non-null   int64
#  3   id         9930 non-null   object
#  4   age        9930 non-null   int64
#  5   male       9930 non-null   int64
#  6   edu        9930 non-null   object
#  7   action     9930 non-null   int64
#  8   intention  9930 non-null   int64
#  9   contact    9930 non-null   int64
#  10  story      9930 non-null   object
#  11  action2    9930 non-null   int64
# dtypes: int64(8), object(4)
# memory usage: 931.1 KB

# (R code 12.13)
fig, axs = plt.subplots(num=1, ncols=3, sharex=True, clear=True)
fig.set_size_inches((12, 5), forward=True)

sts.simplehist(df['response'], color='k', rwidth=0.1, ax=axs[0])
axs[0].set(
    xlabel='response',
    ylabel='Count',
    xlim=(0.5, 7.5)
)

# (R code 12.14)
# discrete proportion of each response value
pr_k = df['response'].value_counts().sort_index() / len(df)

# cumulative proportions
cum_pr_k = pr_k.cumsum()

# log cumulative odds (R code 12.15)
lco = logit(cum_pr_k)

# Plot
axs[1].plot(np.arange(1, 8), cum_pr_k, '-o', c='k', mfc='white', lw=1)
axs[1].set(
    xlabel='response',
    ylabel='cumulative proportion',
    ylim=(-0.05, 1.05)
)

axs[2].plot(np.arange(1, 8), lco, '-o', c='k', mfc='white', lw=1)
axs[2].set(
    xlabel='response',
    ylabel='log cumulative odds'
)
axs[2].locator_params(integer=True)

# Likelihood
# lik = np.diff(np.r_[0, cum_pr_k])  # (7,)

# -----------------------------------------------------------------------------
#         Build a basic model (R code 12.16)
# -----------------------------------------------------------------------------
y = df['response'] - 1  # index values 0 - 6
Km1 = int(y.max() - y.min())

with pm.Model() as model:
    cutpoints = pm.Normal('cutpoints', 0, 1.5, shape=(Km1,),
                          transform=pm.distributions.transforms.ordered,
                          initval=np.linspace(-2, 3, Km1))
    R = pm.OrderedLogistic('R', cutpoints=cutpoints, eta=0, observed=y)
    m12_5 = sts.ulam(data=df)

print('m12.5:')
sts.precis(m12_5)

# Get the cumulative probabilities back
model_p = pd.Series(np.r_[expit(m12_5.coef['cutpoints']), 1])
model_p.index = cum_pr_k.index
print(
    pd.DataFrame(dict(
        cum_pr_k=cum_pr_k,
        model_p=model_p,
        diff=cum_pr_k-model_p
    ))
)

# -----------------------------------------------------------------------------
#         Build the linear model (R code 12.24)
# -----------------------------------------------------------------------------
# NOTE this model takes ~2-3 minutes to sample
with pm.Model() as model:
    A = pm.MutableData('A', df['action'])
    I = pm.MutableData('I', df['intention'])
    C = pm.MutableData('C', df['contact'])
    # Slope priors
    β_A = pm.Normal('β_A', 0, 0.5)
    β_I = pm.Normal('β_I', 0, 0.5)
    β_C = pm.Normal('β_C', 0, 0.5)
    β_IC = pm.Normal('β_IC', 0, 0.5)
    β_IA = pm.Normal('β_IA', 0, 0.5)
    # Interaction coefficient as its own linear model
    B_I = pm.Deterministic('B_I', β_I + β_IA*A + β_IC*C)
    # The linear model
    φ = pm.Deterministic('φ', β_A*A + β_C*C + B_I*I)
    cutpoints = pm.Normal('cutpoints', 0, 1.5, shape=(Km1,),
                          transform=pm.distributions.transforms.ordered,
                          initval=np.linspace(-2, 3, Km1))
    R = pm.OrderedLogistic('R', cutpoints=cutpoints, eta=φ, observed=y)
    m12_6 = sts.ulam(data=df)

print('m12.6:')
sts.precis(m12_6, filter=dict(like='β'))

sts.plot_precis(m12_6, filter=dict(like='β'), fignum=2)

# -----------------------------------------------------------------------------
#         Plot posterior predictive
# -----------------------------------------------------------------------------
post = m12_6.get_samples()

kI = [0, 1]
sample_dims = ('chain', 'draw')

fig, axs = plt.subplots(num=3, nrows=2, ncols=3,
                        sharex='row', sharey='row', clear=True)

for i, (a, c) in enumerate(zip([0, 1, 0],
                               [0, 0, 1])):
    # Sample the link function (mean, φ)
    φ_samp = sts.lmeval(
        m12_6,
        out=m12_6.model.φ,
        eval_at=dict(
            A=np.full_like(kI, a),
            I=kI,
            C=np.full_like(kI, c)
        )
    )

    # # Sample the posterior predictive
    # y_samp = pm.sample_posterior_predictive(post, model=m12_6.model)

    # Compute the probabilities from the intercept and linear model
    pk = expit(post['cutpoints'] - φ_samp)

    q = 0.89
    qq = (1 - q) / 2
    pk_ci = pk.quantile([qq, 1 - qq], sample_dims)

    # Plot the probabilities on the top row
    ax = axs[0, i]
    ax.plot([0, 1], pk.mean(sample_dims).T, c='k', lw=1)  # plot all 6 at once
    for j in range(Km1):
        ax.fill_between([0, 1],
                        pk_ci.isel(quantile=0, cutpoints_dim_0=j),
                        pk_ci.isel(quantile=1, cutpoints_dim_0=j),
                        facecolor='k', alpha=0.3)

    # Get the distribution of responses
    tf = df.loc[(df['action'] == a) & (df['contact'] == c),
                ['intention', 'response']]
    g = tf.groupby('intention')['response'].value_counts().sort_index()

    # Plot the cutpoint data
    pr = g.groupby('intention').transform(lambda x: (x / x.sum()).cumsum())
    for k in kI:
        ax.scatter(np.full(Km1, k), pr.loc[k, :6], c='C0', alpha=0.4)

    # TODO use the "first in column" feature
    if i == 0:
        ax.set(ylabel='probability',
               xticks=(0, 1),
               ylim=(-0.05, 1.05))

    ax.set(title=f"action = {a}, contact = {c}")
    ax.set_xlabel('intention', labelpad=-10)

    # Plot the frequencies on the bottom row
    # NOTE these bars are supposed to be *simulated*, not the data.
    bw = 0.2  # bar width
    responses = g[0].index

    ax = axs[1, i]
    for k, c in zip(kI, ['k', 'C0']):
        ax.bar(responses - (1-k)*bw, g[k], width=bw,
               align='edge', color=c, alpha=0.6, ec='white',
               label=f"intention = {k}")

    if i == 0:
        ax.legend()
        ax.set(ylabel='frequency',
               xticks=responses)

    ax.set(xlabel='response')


# =============================================================================
# =============================================================================
