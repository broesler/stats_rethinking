#!/usr/bin/env python3
# =============================================================================
#     File: chimpanzees.py
#  Created: 2023-12-05 10:51
#   Author: Bernie Roesler
#
"""
§13.3 Multilevel chimpanzees with social choice. Logistic Regression.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy import stats
from scipy.special import expit

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

# -----------------------------------------------------------------------------
#         Create the model
# -----------------------------------------------------------------------------
# m11.4 + cluster for "block" (R code 13.21)
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    block_id = pm.MutableData('block', df['block'] - 1)
    a = pm.Normal('a', 0, 1.5, shape=(len(df['actor'].unique()),))      # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    g = pm.Normal('g', 0, 0.5, shape=(len(df['block'].unique()),))      # (6,)
    p = pm.Deterministic('p', pm.math.invlogit(a[actor] + b[treatment]))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m13_4 = sts.ulam(data=df)

# (R code 13.22)
print('m13.5:')
sts.precis(m13_5)

fig, ax = plt.subplots(num=1, clear=True)
sts.plot_coef_table(coef_table([m13_4], ['block']), ax=ax)

# Model that ignores block (R code 13.23)
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    a = pm.Normal('a', 0, 1.5, shape=(len(df['actor'].unique()),))      # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    p = pm.Deterministic('p', pm.math.invlogit(a[actor] + b[treatment]))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m13_5 = sts.ulam(data=df)

# (R code 13.24)
cmp = sts.compare([m13_4, m13_5], mnames=['m13.4', 'm13.5'], ic='PSIS')
print('ct:')
with pd.option_context('display.precision', 2):
    print(cmp['ct'])


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
