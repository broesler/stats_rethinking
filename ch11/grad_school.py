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

# TODO implement this function
# sts.postcheck(m11_7, fignum=1)

# def postcheck(fit, x=None, q=0.89, fignum=None):
fit = m11_7

y = fit.model.observed_RVs[0].name

# xv = fit.data[x] or yv.index
yv = fit.data[y]

pred = sts.lmeval(
    m11_7,
    out=m11_7.model['p'],
    dist=post,
)

sims = sts.lmeval(
    m11_7,
    out=m11_7.model['admit'],
    params=m11_7.model.free_RVs,  # ignore deterministics
    dist=post,
)

μ = pred.mean('draw')

q = 0.89
a = (1 - q) / 2
μ_PI = pred.quantile([a, 1-a], dim='draw')
y_PI = sims.quantile([a, 1-a], dim='draw')

# if aggregate:
yv /= df['applications']
y_PI = y_PI.values / df['applications'].values

fig = plt.figure(1, clear=True)
ax = fig.add_subplot()
ax.scatter(yv.index, yv, c='C0', label='data')
ax.errorbar(yv.index, μ, yerr=np.abs(μ_PI - μ), 
            ls='none', marker='o', mfc='none', mec='k', label='pred')
ax.scatter(np.tile(yv.index, (2, 1)), y_PI,
           marker='+', c='k', label='y PI')
ax.set(xlabel='case',
       ylabel='admit')

# Connect points in each department
N_depts = len(df['dept'].cat.categories)
x = 1 + 2*np.arange(N_depts)
xv = np.r_[[x, x+1]]
y0 = df.loc[x, 'admit'] / df.loc[x, 'applications']
y1 = df.loc[x+1, 'admit'] / df.loc[x+1, 'applications']
yv = np.r_[[y0, y1]]

ax.plot(xv, yv, 'C0')

# TODO label each line with dept

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
