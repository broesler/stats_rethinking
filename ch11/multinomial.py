#!/usr/bin/env python3
# =============================================================================
#     File: multinomial.py
#  Created: 2023-12-13 12:25
#   Author: Bernie Roesler
#
"""
§11.1.5 Multinomial Regression.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats
from scipy.special import softmax

import stats_rethinking as sts

rng = np.random.default_rng(seed=56)

# ----------------------------------------------------------------------------- 
#         Case 1
# -----------------------------------------------------------------------------
# Simulate career choices among 500 individuals
N = 500                   # number of individuals
income = np.arange(1, 4)  # expected income
score = 0.5 * income      # scores for each career, based on income

# Convert scores to probabilities
p = softmax(score)

# Random choices of careers
careers = rng.choice(income, size=N, p=p)

df = pd.DataFrame(dict(career=careers))

# Fit a model to the data
with pm.Model() as model:
    b = pm.Normal('b', 0, 5)
    s2 = pm.Deterministic('s2', 2*b)
    s3 = pm.Deterministic('s3', 3*b)
    p = pm.Deterministic(
        'p', 
        pm.math.softmax(pm.math.stack([pm.math.zeros_like(s2), s2, s3]))
    )
    career = pm.Categorical('career', p, observed=df['career'])
    # FIXME fails on first iteration
    # m11_9 = sts.ulam(data=df)

# ----------------------------------------------------------------------------- 
#         Case 2
# -----------------------------------------------------------------------------
N = 100

# Simulate family incomes for each individual
family_income = rng.random(N)

# Assign unique coefficient for each type of event
coef = np.r_[1, 0, -1]

score = 0.5*income + coef*family_income[:, np.newaxis]  # (N, 3)
p = softmax(score, axis=1)
careers = np.array([rng.choice(income, p=p[i]) for i, _ in enumerate(p)])

df = pd.DataFrame(dict(career=careers, family_income=family_income))

# Fit a model to the data
with pm.Model() as model:
    a2 = pm.Normal('a2', 0, 5)
    a3 = pm.Normal('a3', 0, 5)
    b2 = pm.Normal('b2', 0, 5)
    b3 = pm.Normal('b3', 0, 5)
    s2 = pm.Deterministic('s2', a2 + b2*family_income)
    s3 = pm.Deterministic('s3', a3 + b3*family_income)
    p = pm.Deterministic(
        'p', 
        pm.math.softmax(pm.math.concatenate([pm.math.zeros_like(s2), s2, s3]))
    )
    career = pm.Categorical('career', p, observed=df['career'])
    # career = pm.Multinomial('career', N, p, observed=df['career'])
    m11_10 = sts.ulam(data=df)

print('m11.10:')
sts.precis(m11_10)




# =============================================================================
# =============================================================================
