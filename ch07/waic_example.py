#!/usr/bin/env python3
# =============================================================================
#     File: waic_example.py
#  Created: 2023-06-27 21:43
#   Author: Bernie Roesler
#
"""
Overthinking: WAIC Calculations (R Code 7.20 - 25)
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

import stats_rethinking as sts

from pathlib import Path
from scipy import stats

data_file = Path('../data/cars.csv')
df = pd.read_csv(data_file, index_col=0)

# Build a linear model of distance ~ speed
with pm.Model():
    a = pm.Normal('a', 0, 100)
    b = pm.Normal('b', 0, 10)
    μ = pm.Deterministic('μ', a + b * df['speed'])
    σ = pm.Exponential('σ', 1)
    d = pm.Normal('d', μ, σ, observed=df['dist'])
    q = sts.quap(data=df)

Ns = 1000
post = q.sample(Ns)

mu_samp = sts.lmeval(q, out=q.model.μ)
ax = sts.lmplot(fit_x=df['speed'], fit_y=mu_samp, data=df, x='speed', y='dist')

d_logp = stats.norm(mu_samp, post['σ']).logpdf(df[['dist']])
lppd = sts.log_sum_exp(d_logp, axis=1) - np.log(Ns)

pWAIC = np.var(d_logp, axis=1)

print(f"WAIC = {-2 * (lppd.sum() - pWAIC.sum()):.4f}")

n_cases = len(df)
waic_vec = -2 * (lppd - pWAIC)
print(f"std(WAIC) = {(n_cases * np.var(waic_vec))**0.5:.4f}")

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
