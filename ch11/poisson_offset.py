#!/usr/bin/env python3
# =============================================================================
#     File: poisson_offset.py
#  Created: 2024-02-13 11:52
#   Author: Bernie Roesler
#
"""
Simulate Poisson model with offset.
"""
# =============================================================================

import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

# Simulate data with daily counts (R code 11.53)
N_days = 30  # 1 month
y_daily = stats.poisson(1.5).rvs(N_days)

# Simulate data with weekly counts (R code 11.54)
N_weeks = 4
days_week = 7
y_weekly = stats.poisson(days_week * 0.5).rvs(N_weeks)

exposure = np.r_[np.ones(N_days), np.repeat(days_week, N_weeks)].astype(int)
monastery = np.r_[np.zeros(N_days), np.ones(N_weeks)].astype(int)

tf = pd.DataFrame(dict(
    y=np.r_[y_daily, y_weekly],
    days=exposure,
    monastery=monastery
))

tf['log_days'] = np.log(tf['days'])

# Fit the model
with pm.Model() as model:
    log_days = pm.ConstantData('log_days', tf['log_days'])
    m = pm.ConstantData('m', tf['monastery'])
    α = pm.Normal('α', 0, 1)
    β = pm.Normal('β', 0, 1)
    λ = pm.Deterministic('λ', pm.math.exp(log_days + α + β*m))
    y = pm.Poisson('y', λ, observed=tf['y'])
    m11_12 = sts.ulam(data=tf)

post = m11_12.get_samples()
λ_daily = np.exp(post['α'])
λ_weekly = np.exp(post['α'] + post['β'])
λ_daily.name = 'λ'
λ_weekly.name = 'λ'

ct = sts.coef_table([λ_daily, λ_weekly], ['daily', 'weekly'], hist=True)
print(ct)

# =============================================================================
# =============================================================================
