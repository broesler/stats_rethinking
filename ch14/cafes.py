#!/usr/bin/env python3
# =============================================================================
#     File: cafes.py
#  Created: 2024-02-19 21:04
#   Author: Bernie Roesler
#
"""
§14.1 Varying Slopes at Cafés.
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

# Simulate the cafes (R code 14.1)
a = 3.5        # avg morning wait time
b = -1         # avg *difference* in afternoon wait time
σ_a = 1    # std in intercepts
σ_b = 0.5  # std in slopes
ρ = -0.7       # correlation between intercepts and slopes

Mu = np.r_[a, b]  # (R code 14.2)

# Directly define covariance matrix (R code 14.3)
cov_ab = σ_a * σ_b * ρ
Σa = np.c_[[σ_a**2, cov_ab],
           [cov_ab, σ_b**2]]

# Define via multiply with correlation matrix (R code 14.5)
sigmas = np.r_[σ_a, σ_b]
Rho = np.c_[[1, ρ],
            [ρ, 1]]
Σ = np.diag(sigmas) @ Rho @ np.diag(sigmas)

np.testing.assert_allclose(Σa, Σ)

# (R code 14.6, 14.7)
N_cafes = 20
rng = np.random.default_rng(seed=56)
vary_effects = stats.multivariate_normal.rvs(
    mean=Mu, cov=Σ, size=N_cafes, random_state=rng
)

# (R code 14.8)
a_cafe, b_cafe = vary_effects.T

# Grid of Gaussian values to plot contours (R code 14.9)
xg, yg = np.mgrid[1:7:0.01, -2.5:0.5:0.01]
zg = stats.multivariate_normal.pdf(np.dstack((xg, yg)), mean=Mu, cov=Σ)
zg = (zg.max() - zg) / zg.max()

# Plot the slopes and intercepts
fig, ax = plt.subplots(num=1, clear=True)
cs = ax.contour(xg, yg, zg, cmap='Blues', levels=[0.1, 0.3, 0.5, 0.8, 0.99])
ax.clabel(cs, cs.levels, inline=True)
ax.scatter(a_cafe, b_cafe, ec='k', fc='none')
ax.set(xlabel='intercepts (a_cafe)',
       ylabel='slopes (b_cafe)')

# (R code 14.10)
N_visits = 10
afternoon = np.repeat([0, 1], N_visits*N_cafes/2)
cafe_id = np.repeat(range(N_cafes), N_visits)
μ = a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon
σ = 0.5  # std *within* a cafe
wait = stats.norm.rvs(loc=μ, scale=σ, size=N_visits*N_cafes)

df = pd.DataFrame(dict(cafe=cafe_id, afternoon=afternoon, wait=wait))

# ----------------------------------------------------------------------------- 
#         Plot correlation density
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(num=2, clear=True)
for η in [1, 2, 4]:
    R = pm.draw(pm.LKJCorr.dist(n=2, eta=η), 10_000)
    sns.kdeplot(R, bw_adjust=0.5, ax=ax)
    # ax.text(s=f"{η = }", x=0.0, y=R.max())

ax.legend()
ax.set(xlabel='correlation',
       ylabel='Density')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
