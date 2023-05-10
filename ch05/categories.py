#!/usr/bin/env python3
# =============================================================================
#     File: categories.py
#  Created: 2023-05-10 12:43
#   Author: Bernie Roesler
#
"""
Description: §5.3 Categorical Variables
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')
rng = np.random.default_rng(seed=63)

# -----------------------------------------------------------------------------
#        Load Dataset (R code 5.34)
# -----------------------------------------------------------------------------
data_path = Path('../data/')
data_file = Path('Howell1.csv')

df = pd.read_csv(data_path / data_file)

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 544 entries, 0 to 543
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   height  544 non-null    float64
#  1   weight  544 non-null    float64
#  2   age     544 non-null    float64
#  3   male    544 non-null    int64     # <-- indicator variable [0, 1]
# dtypes: float64(3), int64(1)
# memory usage: 17.1 KB

# Actual pandas category:
df['male'] = (df['male'].astype('category').cat
                .rename_categories({0: 'female', 1: 'male'})
              )
Nc = len(df['male'].cat.categories)
assert Nc == 2

# Directly simulate priors for the model: (R code 5.35)
#   h ~ N(μ, σ)
#   μ = α + β m
#   α ~ N(178, 20)
#   β ~ N(0, 10)
#   σ ~ U(0, 50)
# where m is an indicator or "dummy" variable.

N = 10_000
mu_female = stats.norm(178, 20).rvs(N)
mu_male = stats.norm(178, 20).rvs(N) + stats.norm(0, 10).rvs(N)
sts.precis(pd.DataFrame({'mu_f': mu_female, 'mu_m': mu_male}))

# Re-cast sex as an "index" variable (indices are 0-index, so no change!)
df['sex'] = df['male'].cat.codes  # (R code 5.26)

# Define a model (R code 5.37)
with pm.Model() as m5_8:
    α = pm.Normal('α', 178, 20, shape=(Nc,))
    μ = pm.Deterministic('μ', α[df['sex']])
    σ = pm.LogNormal('σ', 0, 2)  # Uniform is unstable with pm.find_MAP
    h = pm.Normal('h', μ, σ, observed=df['height'])
    quap5_8 = sts.quap(data=df)

sts.precis(quap5_8)

# Compute a contrast between female and male heights (R code 5.38)
post = quap5_8.sample()
post['diff_fm'] = post['α__0'] - post['α__1']
sts.precis(post)

# -----------------------------------------------------------------------------
#        Load Many Categories Dataset (R code 5.39)
# -----------------------------------------------------------------------------
data_path = Path('../data/')
data_file = Path('milk.csv')

df = pd.read_csv(data_path / data_file)

# >>> df.ino()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 29 entries, 0 to 28
# Data columns (total 8 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   clade           29 non-null     object
#  1   species         29 non-null     object
#  2   kcal.per.g      29 non-null     float64
#  3   perc.fat        29 non-null     float64
#  4   perc.protein    29 non-null     float64
#  5   perc.lactose    29 non-null     float64
#  6   mass            29 non-null     float64
#  7   neocortex.perc  17 non-null     float64
# dtypes: float64(6), object(2)
# memory usage: 1.9+ KB

# Set up integer indices using the categories (R code 5.40)
df['clade'] = df['clade'].astype('category')
df['clade_id'] = df['clade'].cat.codes
Nc = len(df['clade'].cat.categories)

# Create the model (R code 5.41)
df['K'] = sts.standardize(df['kcal.per.g'])

with pm.Model() as m5_9:
    α = pm.Normal('α', 0, 0.5, shape=(Nc,))
    μ = pm.Deterministic('μ', α[df['clade_id']])
    σ = pm.Exponential('σ', 1)
    K = pm.Normal('K', μ, σ, observed=df['K'])
    quap5_9 = sts.quap(data=df)

sts.precis(quap5_9)

# Figure ?? [p 153]
ct = sts.coef_table(models=[quap5_9], mnames=['m5.9'], params=['α'])
sts.plot_coef_table(ct, fignum=1)

# Create an arbitrary new category of houses (R code 5.42)
Nh = 4
df['house_id'] = rng.choice(np.repeat(np.arange(Nh), 2*Nh), size=len(df))

# Two categories in the model (R code 5.43)
with pm.Model() as m5_10:
    α = pm.Normal('α', 0, 0.5, shape=(Nc,))
    h = pm.Normal('h', 0, 0.5, shape=(Nh,))
    μ = pm.Deterministic('μ', α[df['clade_id']] + h[df['house_id']])
    σ = pm.Exponential('σ', 1)
    K = pm.Normal('K', μ, σ, observed=df['K'])
    quap5_10 = sts.quap(data=df)

sts.precis(quap5_10)
ct = sts.coef_table(models=[quap5_10], mnames=['m5.10'], params=['α', 'h'])
sts.plot_coef_table(ct, fignum=2)


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
