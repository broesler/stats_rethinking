#!/usr/bin/env python3
# =============================================================================
#     File: 8H1.py
#  Created: 2023-09-21 08:26
#   Author: Bernie Roesler
#
"""
Chapter 8, Exercises 8H1 and 8H2.
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

df = pd.read_csv(Path('../data/tulips.csv'))

# Normalize variables
df['blooms_std'] = df['blooms'] / df['blooms'].max()
df['water_cent'] = df['water'] - df['water'].mean()
df['shade_cent'] = df['shade'] - df['shade'].mean()

# ----------------------------------------------------------------------------- 
#         8H1. Include `bed` as a predictor
# -----------------------------------------------------------------------------
df['bed_cat'] = df['bed'].astype('category')
df['bed'] = df['bed_cat'].cat.codes  # integers
Ncat = len(df['bed_cat'].cat.categories)

# Add the categorical predictor as a main effect (index variable)
with pm.Model():
    water = pm.MutableData('water', df['water_cent'])
    shade = pm.MutableData('shade', df['shade_cent'])
    bed = pm.MutableData('bed', df['bed'])
    α = pm.Normal('α', 0.5, 0.25, shape=(Ncat,))
    βw = pm.Normal('βw', 0, 0.25, shape=(Ncat,))
    βs = pm.Normal('βs', 0, 0.25, shape=(Ncat,))
    βws = pm.Normal('βws', 0, 0.25, shape=(Ncat,))
    μ = pm.Deterministic(
        'μ', 
        α[bed] + βw[bed]*water + βs[bed]*shade + βws[bed]*water*shade
    )
    σ = pm.Exponential('σ', 1)
    blooms_std = pm.Normal('blooms_std', μ, σ,
                           observed=df['blooms_std'], 
                           shape=water.shape)
    m8_8 = sts.quap(data=df)

print('m8.8:')
sts.precis(m8_8)

# Compare to model 8.7 that omits bed
with pm.Model():
    water = pm.MutableData('water', df['water_cent'])
    shade = pm.MutableData('shade', df['shade_cent'])
    α = pm.Normal('α', 0.5, 0.25)
    βw = pm.Normal('βw', 0, 0.25)
    βs = pm.Normal('βs', 0, 0.25)
    βws = pm.Normal('βws', 0, 0.25)
    μ = pm.Deterministic('μ', α + βw*water + βs*shade + βws*water*shade)
    σ = pm.Exponential('σ', 1)
    blooms_std = pm.Normal('blooms_std', μ, σ, observed=df['blooms_std'])
    m8_7 = sts.quap(data=df)

models = [m8_8, m8_7]
mnames = ['with bed', 'without bed']
cmp = sts.compare(models, mnames)
ct = cmp['ct'].xs('blooms_std')
print(ct)
#                   WAIC         SE     dWAIC        dSE    penalty    weight
# model
# with bed    -16.245141  13.000700  6.714379  12.642001  18.467496  0.033661
# without bed -22.959521   9.977217  0.000000        NaN   6.125850  0.966339

# NOTE the model *without* the categorical variable has a lower WAIC value!
# Upon inspection of the coef table, the posterior distribution of the
# parameters *with* bed included has more uncertainty. This uncertainty
# probably comes from the fact that there are 1/3 as many data points in each
# category as opposed to including all of the data in the model.

sts.plot_coef_table(sts.coef_table(models, mnames))


# =============================================================================
# =============================================================================
