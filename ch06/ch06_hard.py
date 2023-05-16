#!/usr/bin/env python3
# =============================================================================
#     File: ch06_hard.py
#  Created: 2023-05-15 22:13
#   Author: Bernie Roesler
#
"""
Description: Solutions to Chapter 6 "Hard" exercises.

See the DAG 6.2 (R code 6.31, p 184)
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# -----------------------------------------------------------------------------
#        Load Dataset (R code 5.1)
# -----------------------------------------------------------------------------
data_path = Path('../data/')
data_file = Path('WaffleDivorce.csv')

df = pd.read_csv(data_path / data_file)

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 50 entries, 0 to 49
# Data columns (total 15 columns):
#    Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   Location           50 non-null     object
#  1   Loc                50 non-null     object
#  2   Population         50 non-null     float64
#  3   MedianAgeMarriage  50 non-null     float64
#  4   Marriage           50 non-null     float64
#  5   Marriage SE        50 non-null     float64
#  6   Divorce            50 non-null     float64
#  7   Divorce SE         50 non-null     float64
#  8   WaffleHouses       50 non-null     int64
#  9   South              50 non-null     int64
#  10  Slaves1860         50 non-null     int64
#  11  Population1860     50 non-null     int64
#  12  PropSlaves1860     50 non-null     float64
#  13  A                  50 non-null     float64
#  14  D                  50 non-null     float64
# dtypes: float64(9), int64(4), object(2)
# memory usage: 6.0 KB

# Standardize the data
df['A'] = sts.standardize(df['MedianAgeMarriage'])
df['D'] = sts.standardize(df['Divorce'])
df['M'] = sts.standardize(df['Marriage'])
df['W'] = sts.standardize(df['WaffleHouses'])

# Define the model
with pm.Model():
    ind = pm.MutableData('ind', df['W'])
    obs = pm.MutableData('obs', df['D'])
    alpha = pm.Normal('alpha', 0, 0.2)
    beta_W = pm.Normal('beta_W', 0, 0.5)
    sigma = pm.Exponential('sigma', 1.)
    mu = pm.Deterministic('mu', alpha + beta_W*ind)
    D = pm.Normal('D', mu, sigma, observed=obs)
    # Compute the MAP estimate quadratic approximation
    quapW = sts.quap(data=df)

print('W ~ D:')
sts.precis(quapW)

# Define the model
with pm.Model():
    ind = pm.MutableData('ind', df['W'])
    obs = pm.MutableData('obs', df['D'])
    alpha = pm.Normal('alpha', 0, 0.2)
    beta_W = pm.Normal('beta_W', 0, 0.5)
    beta_S = pm.Normal('beta_S', 0, 0.5)
    sigma = pm.Exponential('sigma', 1.)
    mu = pm.Deterministic('mu', alpha + beta_W*ind + beta_S*df['South'])
    D = pm.Normal('D', mu, sigma, observed=obs, shape=ind.shape)
    # Compute the MAP estimate quadratic approximation
    quapS = sts.quap(data=df)

print('W ~ D, controlling for S:')
sts.precis(quapS)

# Define the model
with pm.Model():
    ind = pm.MutableData('ind', df['W'])
    obs = pm.MutableData('obs', df['D'])
    alpha = pm.Normal('alpha', 0, 0.2)
    beta_W = pm.Normal('beta_W', 0, 0.5)
    beta_A = pm.Normal('beta_A', 0, 0.5)
    beta_M = pm.Normal('beta_M', 0, 0.5)
    sigma = pm.Exponential('sigma', 1.)
    mu = pm.Deterministic('mu', alpha 
                                + beta_W*ind + beta_A*df['A'] + beta_M*df['M'])
    D = pm.Normal('D', mu, sigma, observed=obs, shape=ind.shape)
    # Compute the MAP estimate quadratic approximation
    quapAM = sts.quap(data=df)

print('W ~ D, controlling for A and M:')
sts.precis(quapAM)

# =============================================================================
# =============================================================================
