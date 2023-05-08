#!/usr/bin/env python3
# =============================================================================
#     File: spurious_sim.py
#  Created: 2023-05-07 22:39
#   Author: Bernie Roesler
#
"""
Description: Overthinking R code 5.17. Simulating spurious association.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

N = 100  # number of cases
x_real = stats.norm.rvs(size=N)  # N(0, 1)
x_spur = stats.norm(loc=x_real).rvs(N)
y = stats.norm(loc=x_real).rvs(N)
df = pd.DataFrame(np.c_[x_real, x_spur, y], columns=['x_real', 'x_spur', 'y'])

# Model the association
with pm.Model() as multi_model:
    alpha = pm.Normal('alpha', 0, 1)
    beta_r = pm.Normal('beta_r', 0, 1)
    beta_s = pm.Normal('beta_s', 0, 1)
    sigma = pm.Exponential('sigma', 1)
    mu = pm.Deterministic('mu', alpha + beta_r * x_real + beta_s * x_spur)
    y = pm.Normal('y', mu, sigma, observed=y)
    quap = sts.quap()

print('y ~ x_real, x_spur:')
sts.precis(quap)
# y ~ x_real, x_spur:
#           mean    std    5.5%  94.5%
# alpha  -0.1449 0.1493 -0.3835 0.0937
# beta_r  1.0843 0.1164  0.8982 1.2704
# beta_s -0.0652 0.1007 -0.2261 0.0957  <- nearly 0!
# sigma   1.0005 0.0702  0.8883 1.1128

g = sns.PairGrid(df, corner=True)
g.map_diag(sns.kdeplot)
g.map_lower(sns.regplot)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
