#!/usr/bin/env python3
# =============================================================================
#     File: 12H6.py
#  Created: 2024-02-02 11:26
#   Author: Bernie Roesler
#
"""
12H6. Fish data.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy.special import expit

import stats_rethinking as sts

df = pd.read_csv(Path('../data/fish.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 250 entries, 0 to 249
# Data columns (total 6 columns):
#    Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   fish_caught  250 non-null    int64    # Number of fish caught
#  1   livebait     250 non-null    int64    # Whether group used livebait
#  2   camper       250 non-null    int64    # Whether group had a camper
#  3   persons      250 non-null    int64    # Number of adults in group
#  4   child        250 non-null    int64    # Number of children in group
#  5   hours        250 non-null    float64  # Number of hours spent in park
# dtypes: float64(1), int64(5)
# memory usage: 11.8 KB

df['livebait'] = df['livebait'].astype(bool)
df['camper'] = df['camper'].astype(bool)

# Model `fish_caught` as a zero-inflated Poisson.
with pm.Model():
    # Adjust for rate (see p 362, λ = μ/τ)
    τ = pm.ConstantData('τ', df['hours'])
    π = pm.ConstantData('π', df['persons'] + df['child'])
    L = pm.ConstantData('L', df['livebait'].astype(int))
    # Prior probability of going fishing
    α_p = pm.Normal('α_p', 0, 1, shape=(2,))    # => 0.27  < expit(α_p) < 0.73
    # Prior mean fish caught per visitor per hour
    α_m = pm.Normal('α_m', -2, 2, shape=(2,))   # => 0.018 < exp(α_m)   < 1.0
    # Means
    μ = pm.math.exp(α_m[L] + pm.math.log(τ) + pm.math.log(π))
    p = pm.math.invlogit(α_p[L])
    # Predictor
    F = pm.ZeroInflatedPoisson('F', psi=p, mu=μ, observed=df['fish_caught'])
    m_fish = sts.ulam(data=df)

sts.precis(m_fish)

print(f"prob no live bait  = {expit(m_fish.coef['α_p'])[0]:.2f}")
print(f"prob live bait     = {expit(m_fish.coef['α_p'])[1]:.2f}")
print(f"fish no live bait  = {np.exp(m_fish.coef['α_m'][0]):.2f}")
print(f"fish live bait     = {np.exp(m_fish.coef['α_m'][1]):.2f}")

# TODO plot predictions with extra zeros?

# =============================================================================
# =============================================================================
