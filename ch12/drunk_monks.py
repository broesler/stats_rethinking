#!/usr/bin/env python3
# =============================================================================
#     File: drunk_monks.py
#  Created: 2024-01-18 16:56
#   Author: Bernie Roesler
#
"""
§12.2 Zero-Inflated Outcomes.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats
from scipy.special import expit

import stats_rethinking as sts

np.random.seed(365)

# (R code 12.7)
# Define parameters
prob_drink = 0.2  # [%] 20% of days
rate_work = 1.0   # [manuscript/day] average 1 manuscript per day

# sample one year of production
N = 365  # [days]

# simulate days monks drink
drink = stats.bernoulli.rvs(prob_drink, size=N)

# Simulate manuscripts completed
y = (1 - drink) * stats.poisson.rvs(rate_work, size=N)

zeros_drink = np.sum(drink)
zeros_work = np.sum((y == 0) & (drink == 0))
zeros_total = np.sum(y == 0)

np.testing.assert_allclose(zeros_total, zeros_drink + zeros_work)

# (R code 12.8)
# Plot the outcome variable
fig, ax = plt.subplots(num=1, clear=True)
sts.simplehist(y, ax=ax, color='k', rwidth=0.1)
ax.bar(0, zeros_drink, bottom=zeros_work,
       width=0.1, color='C0', label='zeros by drinking')
ax.legend()
ax.set(xlabel='manuscripts completed each day',
       ylabel='number of days')
ax.spines[['top', 'right']].set_visible(False)

# -----------------------------------------------------------------------------
#         Fit the model (R code 12.9)
# -----------------------------------------------------------------------------
with pm.Model():
    α_p = pm.Normal('α_p', -1.5, 1)
    α_l = pm.Normal('α_l', 1, 0.5)
    λ = pm.math.exp(α_l)
    p = pm.math.invlogit(α_p)
    Y = pm.ZeroInflatedPoisson('Y', psi=1-p, mu=λ, observed=y)
    m12_4 = sts.ulam(data=pd.DataFrame(dict(y=y)))

sts.precis(m12_4)

print(f"prob drink  = {expit(m12_4.coef['α_p']):.2f}")
print(f"rate finish = {np.exp(m12_4.coef['α_l']):.2f}")

# =============================================================================
# =============================================================================
