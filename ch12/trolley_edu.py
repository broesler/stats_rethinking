#!/usr/bin/env python3
# =============================================================================
#     File: trolley_edu.py
#  Created: 2024-01-22 12:34
#   Author: Bernie Roesler
#
"""
Description:
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from cycler import cycler
from pathlib import Path
from scipy import stats
from scipy.special import logit, expit

import stats_rethinking as sts

df = pd.read_csv(Path('../data/Trolley.csv'))

# Make a category for education
cats = [
    'Elementary School',
    'Middle School',
    'Some High School',
    'High School Graduate',
    'Some College',
    "Bachelor's Degree",
    "Master's Degree",
    'Graduate Degree',
]

df['edu'] = (
    df['edu']
    .astype('category')
    .cat.reorder_categories(cats, ordered=True)
)

# -----------------------------------------------------------------------------
#         Plot a Dirichlet distribution
# -----------------------------------------------------------------------------
K = 7  # number of responses
Km1 = K - 1
N_lines = 10

delta = stats.dirichlet(alpha=np.full(K, 2)).rvs(N_lines)  # (N_lines, K)

fig, ax = plt.subplots(num=1, clear=True)
ax.set_prop_cycle(cycler(color=plt.cm.viridis.resampled(N_lines).colors))
ax.plot(delta.T, ls='-', marker='o', mfc='white')
ax.set(title=r'Dirichlet$(\alpha=2)$',
       xlabel='index',
       ylabel='probability')

# -----------------------------------------------------------------------------
#         Build the Model
# -----------------------------------------------------------------------------
# NOTE this model takes ~2-3 minutes to sample
with pm.Model() as model:
    A = pm.MutableData('A', df['action'])
    I = pm.MutableData('I', df['intention'])
    C = pm.MutableData('C', df['contact'])
    E = pm.MutableData('E', df['edu'].cat.codes + 1)
    # Slope priors
    β_A = pm.Normal('β_A', 0, 1)
    β_I = pm.Normal('β_I', 0, 1)
    β_C = pm.Normal('β_C', 0, 1)
    β_E = pm.Normal('β_E', 0, 1)
    δ = pm.Dirichlet('δ', np.full(K, 2))
    # Append a 0 and sum each of the predictor rows
    δj = pm.math.concatenate([[0], δ])
    D = pm.Deterministic('D', pm.math.stack([pm.math.sum(δj[:e]) for e in E]))
    # The linear model
    φ = pm.Deterministic('φ', β_A*A + β_I*I + β_C*C + β_E*D)
    κ = pm.Normal('κ', 0, 1.5, shape=(Km1,),
                  transform=pm.distributions.transforms.ordered,
                  initval=np.linspace(-2, 3, Km1))
    R = pm.OrderedLogistic('R', cutpoints=κ, eta=φ, shape=φ.shape,
                           observed=df['response'])
    # m12_7 = sts.ulam(data=df)

# print('m12.7:')
# sts.precis(m12_7, filter=dict(like='β'))

# =============================================================================
# =============================================================================
