#!/usr/bin/env python3
# =============================================================================
#     File: grad_school.py
#  Created: 2023-12-11 13:23
#   Author: Bernie Roesler
#
"""
§11.1.4 Aggregated binomial: Graduate school admissions.
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

df = pd.read_csv(Path('../data/UCBadmit.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# Index: 12 entries, 1 to 12
# Data columns (total 5 columns):
#    Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   dept              12 non-null     object
#  1   applicant.gender  12 non-null     object
#  2   admit             12 non-null     int64
#  3   reject            12 non-null     int64
#  4   applications      12 non-null     int64
# dtypes: int64(3), object(2)
# memory usage: 576.0 bytes

df = df.reset_index().rename({'index': 'case'}, axis='columns')
df['dept'] = df['dept'].astype('category')
df['gender'] = df['applicant.gender'].astype('category')
df = df.drop('applicant.gender', axis='columns')

assert (df['applications'] == df['admit'] + df['reject']).all()

# These are aggregated data.by dept and gender
# N = df['applications'].sum()  # == 4526  # original data rows

df['gid'] = df['gender'].cat.codes

with pm.Model():
    α = pm.Normal('α', 0, 1.5, shape=(2,))
    p = pm.Deterministic('p', pm.math.invlogit(α[df['gid']]))
    admit = pm.Binomial('admit', df['applications'], p, observed=df['admit'])
    m11_7 = sts.ulam(data=df)

print('m11.7:')
sts.precis(m11_7)

post = m11_7.get_samples()
diff_a = post['α'].diff('α_dim_0').squeeze()
diff_p = expit(post['α']).diff('α_dim_0').squeeze()
sts.precis(xr.Dataset(dict(diff_a=diff_a, diff_p=diff_p)))

df['admit_p'] = df['admit'] / df['applications']


# TODO move into stats_rethinking
def postcheck(fit, agg_name=None, N=1000, q=0.89, fignum=None):
    """Plot the discrete observed data and the posterior predictions.

    Parameters
    ----------
    fit : :obj:`sts.PostModel'
        The model to which the data is fitted.
    agg_name : str, optional
        The name of the variable over which the data is aggregated.
    N : int, optional
        The number of samples to take of the posterior.
    q : float in [0, 1], optional
        The quantile of which to compute the interval.
    fignum : int, optional
        The Figure number in which to plot.

    Returns
    -------
    ax : plt.Axes
        The axes in which the plot was drawn.
    """
    y = fit.model.observed_RVs[0].name
    post = fit.get_samples(N)

    yv = fit.data[y].copy()
    xv = np.arange(len(yv))

    pred = sts.lmeval(
        fit,
        out=fit.model['p'],
        dist=post,
    )

    sims = sts.lmeval(
        fit,
        out=fit.model[y],
        params=fit.model.free_RVs,  # ignore deterministics
        dist=post,
    )

    μ = pred.mean('draw')

    a = (1 - q) / 2
    μ_PI = pred.quantile([a, 1-a], dim='draw')
    y_PI = sims.quantile([a, 1-a], dim='draw')

    if agg_name is not None:
        yv /= fit.data[agg_name]
        y_PI = y_PI.values / fit.data[agg_name].values

    fig = plt.figure(fignum, clear=True)
    ax = fig.add_subplot()

    # Plot the mean and simulated PIs
    ax.errorbar(xv, μ, yerr=np.abs(μ_PI - μ), c='k',
                ls='none', marker='o', mfc='none', mec='k', label='pred')
    ax.scatter(np.tile(xv, (2, 1)), y_PI, marker='+', c='k', label='y PI')
    # Plot the data
    ax.scatter(xv, yv, c='C0', label='data', zorder=10)

    ax.legend()
    ax.set(xlabel='case',
        ylabel=y)

    ax.spines[['top', 'right']].set_visible(False)
    return ax


# (R code 11.31)
# sts.postcheck(m11_7, fignum=1)
ax = postcheck(m11_7, agg_name='applications', fignum=1)

# TODO can maybe generalize the labeling and line grouping given up to 2 levels
# of grouping. 
# Need to compute the center of each group for the major label and individual
# point locations for the minor label.
# Maybe take kwargs "major_group='dept', minor_group='gender'"

# Connect points in each department
N_depts = len(df['dept'].cat.categories)
x = 2*np.arange(N_depts)
xp = np.r_[[x, x+1]]
y0 = df.loc[x, 'admit'] / df.loc[x, 'applications']
y1 = df.loc[x+1, 'admit'] / df.loc[x+1, 'applications']
yp = np.r_[[y0, y1]]

ax.plot(xp, yp, 'C0')

# Label the cases
xv = np.arange(len(df))
xind = xv[:-1:2]

ax.set_xticks(xind + 0.5)
ax.set_xticklabels(df.loc[xind, 'dept'])

ax.set_xticks(xv, minor=True)
ax.set_xticklabels(df['gender'], minor=True)
ax.tick_params(axis='x', which='minor', pad=18)

ax.tick_params(axis='x', which='major',  bottom=False)  # don't draw the ticks
ax.tick_params(axis='x', which='minor',  bottom=True)

# Lines between each department for clarity
for x in xind + 1.5:
    ax.axvline(x, lw=1, c='k')


# Create new model with departments separately indexed (R code 11.32)
with pm.Model():
    α = pm.Normal('α', 0, 1.5, shape=(2,))
    δ = pm.Normal('δ', 0, 1.5, shape=(N_depts,))
    p = pm.Deterministic('p', pm.math.invlogit(α[df['gid']] + δ[df['dept'].cat.codes]))
    admit = pm.Binomial('admit', df['applications'], p, observed=df['admit'])
    m11_8 = sts.ulam(data=df)

print('m11.8:')
sts.precis(m11_8)

# (R code 11.33) Compute the contrasts
post = m11_8.get_samples()
diff_a = post['α'].diff('α_dim_0').squeeze()
diff_p = expit(post['α']).diff('α_dim_0').squeeze()
sts.precis(xr.Dataset(dict(diff_a=diff_a, diff_p=diff_p)))

# (R code 11.34) Tabulate rates of admission across departments
df['applications_p'] = (
    df.groupby('dept', observed=True)
    ['applications'].transform(lambda group: group / group.sum())
)

# Make the pivot table for easy reading
pg = (
    df[['dept', 'gender', 'applications_p']]
    .set_index('gender')
    .pivot(columns=['dept'])
)

print("pg:")
print(pg)


# NOTE why can't we just apply the transformation to one column of groupby!?
# Want to be able to do something like:
# pg = (
#     df
#     .groupby('dept')
#     .transform({
#         'applications': lambda x: x / x.sum(),
#     })
#     [['gender', 'applications']]
# )

# Convoluted workaround using `apply`:
# def f(group):
#     return pd.DataFrame({
#         'gender': group['gender'],
#         'applications': group['applications'] / group['applications'].sum(),
#     })

# pg = (
#     df.groupby('dept')
#     [['gender', 'applications']]
#     .apply(f)
#     .reset_index(level=1, drop=True)
#     .set_index('gender', append=True)
# )


print("pg:")
print(pg)

ax = postcheck(m11_8, agg_name='applications', fignum=2)

x = 2*np.arange(N_depts)
xp = np.r_[[x, x+1]]
y0 = df.loc[x, 'admit'] / df.loc[x, 'applications']
y1 = df.loc[x+1, 'admit'] / df.loc[x+1, 'applications']
yp = np.r_[[y0, y1]]

ax.plot(xp, yp, 'C0')

# Label the cases
xv = np.arange(len(df))
xind = xv[:-1:2]

ax.set_xticks(xind + 0.5)
ax.set_xticklabels(df.loc[xind, 'dept'])

ax.set_xticks(xv, minor=True)
ax.set_xticklabels(df['gender'], minor=True)
ax.tick_params(axis='x', which='minor', pad=18)

ax.tick_params(axis='x', which='major',  bottom=False)  # don't draw the ticks
ax.tick_params(axis='x', which='minor',  bottom=True)

# Lines between each department for clarity
for x in xind + 1.5:
    ax.axvline(x, lw=1, c='k')


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
