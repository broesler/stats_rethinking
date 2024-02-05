#!/usr/bin/env python3
# =============================================================================
#     File: trolley_gender.py
#  Created: 2024-01-24 17:12
#   Author: Bernie Roesler
#
"""
§12.4 Trolley problem with gender, etc. as predictors. See also "11H5".

See 2023 Rethinking Lecture 11 @ 55:16.

We cannot interpret the effect of education, β_E, without controlling for
*both* age and gender! Age and gender "cause" education, which provides
a backdoor to confounding the effect of education on response.
"""
# =============================================================================

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

df = pd.read_csv(
    Path('../data/Trolley.csv'),
    dtype=dict(
        response='category',
        edu='category',
    )
)

# Make a category for education
edu_cats = [
    'Elementary School',
    'Middle School',
    'Some High School',
    'High School Graduate',
    'Some College',
    "Bachelor's Degree",
    "Master's Degree",
    'Graduate Degree',
]

df['edu'] = df['edu'].cat.reorder_categories(edu_cats, ordered=True)

K = df['response'].cat.categories.size  # number of responses
Km1 = K - 1

# -----------------------------------------------------------------------------
#         Build the Model
# -----------------------------------------------------------------------------
# NOTE this model takes ~4 minutes to sample
with pm.Model() as model:
    A = pm.MutableData('A', df['action'])
    I = pm.MutableData('I', df['intention'])
    C = pm.MutableData('C', df['contact'])
    E = pm.MutableData('E', df['edu'].cat.codes)
    G = pm.MutableData('G', df['male'])  # 1 == male, 0 == female
    Y = pm.MutableData('Y', df['age'])   # include age as continuous
    y = df['response'].cat.codes  # must be on [0, 7] scale
    # Slope priors
    β_A = pm.Normal('β_A', 0, 1, shape=(2,))
    β_I = pm.Normal('β_I', 0, 1, shape=(2,))
    β_C = pm.Normal('β_C', 0, 1, shape=(2,))
    β_E = pm.Normal('β_E', 0, 1, shape=(2,))
    β_Y = pm.Normal('β_Y', 0, 1, shape=(2,))
    δ = pm.Dirichlet('δ', np.full(K, 2))  # constrain to a simplex
    # Append a 0 and sum each of the predictor rows.
    # Note that sum(δj[:E+1]) == cumsum(δj)[E]
    # It is MUCH faster to index into a constant array than to evaluate an
    # array of tensor functions.
    δj = pm.math.concatenate([[0], δ])
    D = pm.Deterministic('D', pm.math.cumsum(δj))
    # The linear model
    φ = pm.Deterministic('φ', (β_A[G]*A + β_I[G]*I + β_C[G]*C
                               + β_E[G]*D[E] + β_Y[G]*Y))
    # The cutpoints, constrained to be ordered.
    κ = pm.Normal('κ', 0, 1.5, shape=(Km1,),
                  transform=pm.distributions.transforms.ordered,
                  initval=np.linspace(-2, 3, Km1))
    R = pm.OrderedLogistic('R', cutpoints=κ, eta=φ, shape=φ.shape, observed=y)
    mRXE = sts.ulam(data=df, nuts_sampler='numpyro')

print('mRXE:')
sts.precis(mRXE, filter_kws=dict(regex='β|δ'))

# Are women more or less bothered by contact than men?
#   => Compare β_C[0] and β_C[1].
ct = sts.coef_table([mRXE], ['mRXE'], params=['β_C'])
ct = ct.rename(index={
    'β_C[0]': 'β_C[female]',
    'β_C[1]': 'β_C[male]'
})

sts.plot_coef_table(ct, fignum=1)

# => β_C[male] > β_C[female]! So males actually "disapprove" of contact more
# than females.

# =============================================================================
# =============================================================================
