#!/usr/bin/env python3
# =============================================================================
#     File: africa_mcmc.py
#  Created: 2023-12-01 11:08
#   Author: Bernie Roesler
#
"""
§9.4 Conditional modeling of terrain ruggedness vs GDP... using MCMC sampling.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# Get the data (R code 9.9)
df = pd.read_csv(Path('../data/rugged.csv'))

# >>> df.info(){{{
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 234 entries, 0 to 233
# Data columns (total 51 columns):
#    Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   isocode                 234 non-null    object
#  1   isonum                  234 non-null    int64
#  2   country                 234 non-null    object
#  3   rugged                  234 non-null    float64
#  4   rugged_popw             234 non-null    float64
#  5   rugged_slope            234 non-null    float64
#  6   rugged_lsd              234 non-null    float64
#  7   rugged_pc               234 non-null    float64
#  8   land_area               230 non-null    float64
#  9   lat                     234 non-null    float64
#  10  lon                     234 non-null    float64
#  11  soil                    225 non-null    float64
#  12  desert                  234 non-null    float64
#  13  tropical                234 non-null    float64
#  14  dist_coast              234 non-null    float64
#  15  near_coast              234 non-null    float64
#  16  gemstones               234 non-null    int64
#  17  rgdppc_2000             170 non-null    float64
#  18  rgdppc_1950_m           137 non-null    float64
#  19  rgdppc_1975_m           137 non-null    float64
#  20  rgdppc_2000_m           159 non-null    float64
#  21  rgdppc_1950_2000_m      137 non-null    float64
#  22  q_rule_law              197 non-null    float64
#  23  cont_africa             234 non-null    int64
#  24  cont_asia               234 non-null    int64
#  25  cont_europe             234 non-null    int64
#  26  cont_oceania            234 non-null    int64
#  27  cont_north_america      234 non-null    int64
#  28  cont_south_america      234 non-null    int64
#  29  legor_gbr               211 non-null    float64
#  30  legor_fra               211 non-null    float64
#  31  legor_soc               211 non-null    float64
#  32  legor_deu               211 non-null    float64
#  33  legor_sca               211 non-null    float64
#  34  colony_esp              234 non-null    int64
#  35  colony_gbr              234 non-null    int64
#  36  colony_fra              234 non-null    int64
#  37  colony_prt              234 non-null    int64
#  38  colony_oeu              234 non-null    int64
#  39  africa_region_n         234 non-null    int64
#  40  africa_region_s         234 non-null    int64
#  41  africa_region_w         234 non-null    int64
#  42  africa_region_e         234 non-null    int64
#  43  africa_region_c         234 non-null    int64
#  44  slave_exports           234 non-null    float64
#  45  dist_slavemkt_atlantic  57 non-null     float64
#  46  dist_slavemkt_indian    57 non-null     float64
#  47  dist_slavemkt_saharan   57 non-null     float64
#  48  dist_slavemkt_redsea    57 non-null     float64
#  49  pop_1400                201 non-null    float64
#  50  european_descent        165 non-null    float64
# dtypes: float64(31), int64(18), object(2)
# memory usage: 93.4 KB}}}

# Log version of the output
df['log_GDP'] = np.log(df['rgdppc_2000'])

# Extract countries with GDP data
df = df.dropna(subset='rgdppc_2000')

# Rescale variables
df['log_GDP_std'] = df['log_GDP'] / df['log_GDP'].mean()  # proportion of avg
df['rugged_std'] = df['rugged'] / df['rugged'].max()      # [0, 1]
df['cid'] = (df['cont_africa'] == 1).astype(int)

# Define the quap model (R code 9.10)
with pm.Model() as the_model:
    ind = pm.MutableData('ind', df['rugged_std'])
    obs = pm.MutableData('obs', df['log_GDP_std'])
    cid = pm.MutableData('cid', df['cid'])
    α = pm.Normal('α', 1, 0.1, shape=(2,))
    β = pm.Normal('β', 0, 0.3, shape=(2,))
    μ = pm.Deterministic('μ', α[cid] + β[cid]*(ind - df['rugged_std'].mean()))
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=obs, shape=ind.shape)
    m8_5 = sts.quap(data=df)

print('m8.5:')
sts.precis(m8_5)
#         mean    std    5.5%   94.5%
# α__0  1.0506 0.0099  1.0347  1.0665
# α__1  0.8866 0.0157  0.8615  0.9116
# β__0 -0.1426 0.0547 -0.2301 -0.0551
# β__1  0.1325 0.0742  0.0139  0.2511  # slope reversed in Africa!
# σ     0.1095 0.0059  0.1000  0.1190

# Compute MCMC samples of the posterior
m9_1 = sts.ulam(model=the_model, data=df, chains=4, cores=4)

print('m9.1:')
sts.precis(m9_1)

# m9_1.plot_trace()
# m9_1.pairplot()

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
