#!/usr/bin/env python3
# =============================================================================
#     File: africa.py
#  Created: 2023-09-18 17:59
#   Author: Bernie Roesler
#
"""
§8.1 Conditional modeling of terrain ruggedness vs GDP.
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

# R code 8.1
df = pd.read_csv(Path('../data/rugged.csv'))

# >>> df.info()
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
# memory usage: 93.4 KB

# Log version of the output
df['log_GDP'] = np.log(df['rgdppc_2000'])

# Extract countries with GDP data
df = df.dropna(subset='rgdppc_2000')

# Rescale variables
df['log_GDP_std'] = df['log_GDP'] / df['log_GDP'].mean()  # proportion of avg
df['rugged_std'] = df['rugged'] / df['rugged'].max()      # [0, 1]

# Split into African and non-African countries
dA1 = df.loc[df['cont_africa'] == 1]
dA0 = df.loc[df['cont_africa'] == 0]


def gdp_model(x='rugged_std', y='log_GDP_std', data=df):
    """Create a model of log GDP vs ruggedness of terrain."""
    with pm.Model():
        ind = pm.MutableData('ind', data[x])
        obs = pm.MutableData('obs', data[y])
        a = pm.Normal('a', 1, 1)
        b = pm.Normal('b', 0, 1)
        μ = pm.Deterministic('μ', a + b * (ind - data[x].mean()))
        σ = pm.Exponential('σ', 1)
        y = pm.Normal('y', μ, σ, observed=obs)
        return sts.quap(data=data)

# Model of African countries
m8_1 = gdp_model(data=dA1)

plt.ion()
plt.show()
# =============================================================================
# =============================================================================
