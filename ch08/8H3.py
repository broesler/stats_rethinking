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

# Drop Seychelles from a version of the dataset
df_ns = df.loc[df['country'] != 'Seychelles'].copy()


def build_model(data):
    """Fit the interaction model.

    .. math::
        y ~ N(μ, σ)
        μ ~ α + β_A A + β_R R + β_AR A R

    where A = `cont_africa`, R = `rugged`.
    """
    with pm.Model():
        R = pm.MutableData('R', data['rugged_std'])
        A = pm.MutableData('A', data['cont_africa'])  # index value
        obs = pm.MutableData('obs', data['log_GDP_std'])
        α = pm.Normal('α', 1, 0.1, shape=(2,))
        β = pm.Normal('β', 0, 0.3, shape=(2,))
        μ = pm.Deterministic('μ', α[A] + β[A]*R)
        σ = pm.Exponential('σ', 1)
        y = pm.Normal('y', μ, σ, observed=obs, shape=R.shape)
        return sts.quap(data=data)


# Fit with and without Seychelles
q_all = build_model(df)
q_ns = build_model(df_ns)

print(f"{float(q_all.coef['β'][1] / q_ns.coef['β'][1]) = :.2f}")  # ≈ 2.35

# We now have 4 options: (Non-)?African countries, and with(out)? Seychelles.
ct = sts.coef_table([q_all, q_ns], ['all', 'no Seychelles'])
sts.plot_coef_table(ct, fignum=1)

# =============================================================================
# =============================================================================
