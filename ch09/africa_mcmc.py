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
