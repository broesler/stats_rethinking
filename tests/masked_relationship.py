#!/usr/bin/env python3
# =============================================================================
#     File: masked_relationship.py
#  Created: 2023-05-10 09:23
#   Author: Bernie Roesler
#
"""
Description: Simulate a masked relationship.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')


def build_linear_model(data, x, y, x1=None):
    with pm.Model() as model:
        x0 = pm.MutableData('x0', data[x])
        obs = pm.MutableData('obs', data[y])
        α = pm.Normal('α', 0, 0.2)
        β = pm.Normal('β', 0, 0.5)
        if x1 is None:
            μ = pm.Deterministic('μ', α + β * x0)
        else:
            x1 = pm.MutableData('x1', data[x1])
            β_1 = pm.Normal('β_1', 0, 0.5)
            μ = pm.Deterministic('μ', α + β * x0 + β_1 * x1)
        σ = pm.Exponential('σ', 1)
        K = pm.Normal('K', μ, σ, observed=obs, shape=x0.shape)
        quap= sts.quap()  # compute the posterior
    return quap


Ns = 100  # data points

# M -> K <- N
# M -> N
M = stats.norm.rvs(size=Ns)
N = stats.norm.rvs(size=Ns, loc=M)
K = stats.norm.rvs(size=Ns, loc=N - M)
df0 = pd.DataFrame({'M': M, 'N': N, 'K': K})
m5_5 = build_linear_model(df0, x='N', y='K')
m5_6 = build_linear_model(df0, x='M', y='K')
m5_7 = build_linear_model(df0, x='M', x1='N', y='K')
print('M -> N:')
print(sts.coef_table(
    models=[m5_5, m5_6, m5_7],
    mnames=['m5.5', 'm5.6', 'm5.7'],
    params=['β', 'β_1']
))

# M -> K <- N
# N -> M
N = stats.norm.rvs(size=Ns)
M = stats.norm.rvs(size=Ns, loc=N)
K = stats.norm.rvs(size=Ns, loc=N - M)
df1 = pd.DataFrame({'M': M, 'N': N, 'K': K})
m5_5 = build_linear_model(df1, x='N', y='K')
m5_6 = build_linear_model(df1, x='M', y='K')
m5_7 = build_linear_model(df1, x='M', x1='N', y='K')
print('N -> M:')
print(sts.coef_table(
    models=[m5_5, m5_6, m5_7],
    mnames=['m5.5', 'm5.6', 'm5.7'],
    params=['β', 'β_1']
))

# M -> K <- N
# M <- U -> N
U = stats.norm.rvs(size=Ns)  # unobserved variable
M = stats.norm.rvs(size=Ns, loc=U)
N = stats.norm.rvs(size=Ns, loc=U)
K = stats.norm.rvs(size=Ns, loc=N - M)
df2 = pd.DataFrame({'M': M, 'N': N, 'K': K})
m5_5 = build_linear_model(df2, x='N', y='K')
m5_6 = build_linear_model(df2, x='M', y='K')
m5_7 = build_linear_model(df2, x='M', x1='N', y='K')
print('M <- U -> N:')
print(sts.coef_table(
    models=[m5_5, m5_6, m5_7],
    mnames=['m5.5', 'm5.6', 'm5.7'],
    params=['β', 'β_1']
))


# =============================================================================
# =============================================================================
