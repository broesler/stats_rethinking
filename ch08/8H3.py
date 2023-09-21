#!/usr/bin/env python3
# =============================================================================
#     File: 8H3.py
#  Created: 2023-09-21 11:12
#   Author: Bernie Roesler
#
"""
Chapter 8, Exercise 8H3.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

df = pd.read_csv(Path('../data/rugged.csv'))

# Remove NaNs
df = df.dropna(subset='rgdppc_2000')

# Normalize variables
df['rugged_std'] = df['rugged'] / df['rugged'].max()      # [0, 1]

df['log_GDP'] = np.log(df['rgdppc_2000'])
df['log_GDP_std'] = df['log_GDP'] / df['log_GDP'].mean()  # proportion of avg

# Fit the interaction model:
#   y ~ N(μ, σ)
#   μ ~ α + β_A A + β_R R + β_AR A R
#   A: cont_africa, R: rugged

# TODO fit with and without Seychelles
# Drop Seychelles from the dataset
df = df.loc[df['country'] != 'Seychelles']

with pm.Model():
    R = pm.MutableData('R', df['rugged_std'])
    A = pm.MutableData('A', df['cont_africa'])  # indicator
    obs = pm.MutableData('obs', df['log_GDP_std'])
    α = pm.Normal('α', 1, 0.1, shape=(2,))
    β = pm.Normal('β', 0, 0.3, shape=(2,))
    μ = pm.Deterministic('μ', α[A] + β[A]*R)
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=obs, shape=R.shape)
    m8H3 = sts.quap(data=df)

sts.precis(m8H3)

# =============================================================================
# =============================================================================
