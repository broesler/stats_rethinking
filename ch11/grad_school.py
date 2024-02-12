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
import seaborn as sns
import xarray as xr

from pathlib import Path
from scipy.special import expit

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

# (R code 11.31)
ax = sts.postcheck(
    m11_7,
    mean_name='p',
    agg_name='applications',
    major_group='dept',
    minor_group='gender',
    fignum=1
)
ax.set_title('Model by Gender Only')

N_depts = len(df['dept'].cat.categories)

# Create new model with departments separately indexed (R code 11.32)
with pm.Model():
    α = pm.Normal('α', 0, 1.5, shape=(2,))
    δ = pm.Normal('δ', 0, 1.5, shape=(N_depts,))
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(α[df['gid']] + δ[df['dept'].cat.codes])
    )
    admit = pm.Binomial('admit', df['applications'], p, observed=df['admit'])
    m11_8 = sts.ulam(data=df)

print('m11.8:')
sts.precis(m11_8)

# (R code 11.33) Compute the contrasts by department
post_D = m11_8.get_samples()
diff_aD = post_D['α'].diff('α_dim_0').squeeze()
diff_pD = expit(post_D['α']).diff('α_dim_0').squeeze()
sts.precis(xr.Dataset(dict(diff_aD=diff_aD, diff_pD=diff_pD)))

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

# Re-plot with better predictions
ax = sts.postcheck(
    m11_8,
    mean_name='p',
    agg_name='applications',
    major_group='dept',
    minor_group='gender',
    fignum=2
)
ax.set_title('Model by Department and Gender')


# -----------------------------------------------------------------------------
#         Lecture 9 Model and Plots
# -----------------------------------------------------------------------------
# Follow Lecture 09 at 1:02:38 to model total and direct effects of gender
# and department:
# <https://youtu.be/Zi6N3GLUJmw?si=4sSyaNhENICrtRJE&t=3758>
# Just need to index α[G, D] instead of a separate variable δ.
#
# Could also just add the effects from m11_8 model? should we be
# computing diff_a = α + δ?

# Create new model with departments separately indexed (R code 11.32)
# NOTE this model provides identical predictions to m11_8 above, but is
# properly defined with a single intercept for each case.
with pm.Model():
    N = pm.MutableData('N', df['applications'])
    G = pm.MutableData('G', df['gid'])
    D = pm.MutableData('D', df['dept'].cat.codes)
    A = pm.MutableData('A', df['admit'])
    α = pm.Normal('α', 0, 1.5, shape=(2, N_depts))
    p = pm.Deterministic('p', pm.math.invlogit(α[G, D]))
    admit = pm.Binomial('admit', N, p, shape=p.shape, observed=A)
    mGD = sts.ulam(data=df)

print('mGD:')
sts.precis(mGD)

# Compute the contrasts
post_GD = mGD.get_samples()
diff_pGD = expit(post_GD['α']).diff('α_dim_0').squeeze()

# Plot the distribution of the gender contrast for each model
fig, axs = plt.subplots(num=3, ncols=2, clear=True)
fig.set_size_inches((10, 4), forward=True)

# Gender only
axs[0].set(title='Gender Only',
           xlabel='Gender Contrast [probability]')
sns.kdeplot(diff_p.stack(sample=('chain', 'draw')),
            bw_adjust=0.5, c='C3', ax=axs[0])
axs[0].text(s='men advantaged →', x=0.5, y=0.99,
            ha='center', va='top', transform=axs[0].transAxes)

# Gender and Dept
axs[1].set(title='Gender and Department',
           xlabel='Gender Contrast [probability]')
axs[1].axvline(0, ls='--', c='grey', lw=1, alpha=0.5, zorder=0)
sns.kdeplot(diff_pGD.stack(sample=('chain', 'draw')).transpose('sample', ...),
            bw_adjust=0.5, ax=axs[1], legend=True)
axs[1].text(s='← women advantaged       men advantaged →', x=0.0, y=3.95,
            ha='center', va='top')


# -----------------------------------------------------------------------------
#         Post-stratification Simulation
# -----------------------------------------------------------------------------
total_apps = df['applications'].sum()
apps_per_dept = df.groupby('dept', observed=True)['applications'].sum()

# Simulate if all applications are from women
eval_at = dict(
    D=np.repeat(df['dept'].cat.codes.unique(), apps_per_dept),
    N=np.ones(total_apps).astype(int),
    G=np.zeros(total_apps).astype(int)  # all women
)

p_W = sts.lmeval(mGD, out=mGD.model.p, eval_at=eval_at)

# Simulate if all applications are from men
eval_at['G'] = np.ones(total_apps).astype(int)  # all men
p_M = sts.lmeval(mGD, out=mGD.model.p, eval_at=eval_at)

# Summarize the contrast
diff_p_sim = p_M - p_W

fig, axs = plt.subplots(num=4, ncols=2, clear=True)
fig.set_size_inches((10, 4), forward=True)

ax = axs[0]
ax.axvline(0, ls='--', c='grey', lw=1, alpha=0.5, zorder=0)
sns.kdeplot(diff_p_sim.to_numpy().flatten(), bw_adjust=0.5, ax=ax)
ax.text(s='← women advantaged       men advantaged →', x=0.0, y=8.5,
        ha='center', va='top')
ax.set(title='Post-Stratification',
       xlabel='Effect of Gender Perception')

# Show each dept with weight as in population
w = df.groupby('dept')['applications'].sum() / total_apps

ax = axs[1]
ax.set(title='Post-Stratification by Dept.',
       xlabel='Effect of Gender Perception')
ax.axvline(0, ls='--', c='grey', lw=1, alpha=0.5, zorder=0)
sns.kdeplot(diff_pGD.stack(sample=('chain', 'draw')).transpose('sample', ...),
            bw_adjust=0.5,
            color=(w.values*np.ones((3, 1))).T,  # FIXME doesn't work??
            ax=ax)
ax.text(s='← women advantaged       men advantaged →', x=0.0, y=3.95,
            ha='center', va='top')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
