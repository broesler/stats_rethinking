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
import pandas as pd
import pymc as pm
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

# Re-plot with better predictions
ax = sts.postcheck(
    m11_8,
    mean_name='p',
    agg_name='applications',
    major_group='dept',
    minor_group='gender',
    fignum=2
)


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
