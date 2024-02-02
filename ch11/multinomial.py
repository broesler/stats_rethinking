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

import numpy as np
import pandas as pd
import pymc as pm

from scipy.special import softmax

import stats_rethinking as sts

rng = np.random.default_rng(seed=56)

# -----------------------------------------------------------------------------
#         Case 1
# -----------------------------------------------------------------------------
# Simulate career choices among 500 individuals
N = 500                # number of individuals
income = np.arange(3)  # expected income
score = 0.5 * income   # scores for each career, based on income

# Convert scores to probabilities
probs = softmax(score)

# Random choices of careers
careers = rng.choice(income, size=N, p=probs)

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
    career = pm.Categorical('career', p=p, observed=df['career'])
    m11_9 = sts.ulam(data=df)

print('m11.9:')
sts.precis(m11_9)

idata_9 = pm.sample_posterior_predictive(
    model=m11_9.model,
    trace=m11_9.samples
)
print(idata_9.posterior_predictive['career'])

# -----------------------------------------------------------------------------
#         Case 2
# -----------------------------------------------------------------------------
N = 100  # number of individuals

# Simulate family incomes for each individual
family_income = rng.random(N)

# Assign unique coefficient for each type of event
coef = np.r_[1, 0, -1]
score = 0.5*income + coef*family_income[:, np.newaxis]  # (N, 3)
probs = softmax(score, axis=1)  # (N, 3), rows sum to 1.0
careers = np.array([rng.choice(income, p=x) for x in probs])

df = pd.DataFrame(dict(career=careers, family_income=family_income))

# Fit a model to the data
with pm.Model() as model:
    a2 = pm.Normal('a2', 0, 5)
    a3 = pm.Normal('a3', 0, 5)
    b2 = pm.Normal('b2', 0, 5)
    b3 = pm.Normal('b3', 0, 5)
    s2 = a2 + b2*family_income
    s3 = a3 + b3*family_income
    zero = pm.math.zeros_like(s2)
    score = pm.math.stack([zero, s2, s3], axis=1)
    # stack -> (N, 3), p.sum(axis=1) = 1.0
    p = pm.Deterministic('p', pm.math.softmax(score, axis=1))
    career = pm.Categorical('career', p, observed=df['career'])
    m11_10 = sts.ulam(data=df)

print('m11.10:')
sts.precis(m11_10)

idata_10 = pm.sample_posterior_predictive(
    model=m11_10.model,
    trace=m11_10.samples
)
print(idata_10.posterior_predictive['career'])


# =============================================================================
# =============================================================================
