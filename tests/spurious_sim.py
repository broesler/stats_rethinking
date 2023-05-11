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

plt.close('all')
plt.style.use('seaborn-v0_8-darkgrid')

# Generate spurious correlation data. (R code 5.10 and 5.17)
N = 100  # number of cases
x_real = stats.norm.rvs(size=N)  # N(0, 1)
x_spur = stats.norm(loc=x_real).rvs(N)
y = stats.norm(loc=x_real).rvs(N)
df = pd.DataFrame(np.c_[x_real, x_spur, y], columns=['x_real', 'x_spur', 'y'])

# Explore the data
g = sns.PairGrid(df, corner=True)
g.map_diag(sns.histplot)
g.map_lower(sns.regplot)


# Model the association
def build_linear_model(data, x, y, x1=None):
    with pm.Model() as model:
        x0 = pm.MutableData('x0', data[x])
        obs = pm.MutableData('obs', data[y])
        α = pm.Normal('α', 0, 1)
        β = pm.Normal('β', 0, 1)
        if x1 is None:
            μ = pm.Deterministic('mu', α + β * x0)
        else:
            x1 = pm.MutableData('x1', data[x1])
            β_1 = pm.Normal('β_1', 0, 1)
            μ = pm.Deterministic('mu', α + β * x0 + β_1 * x1)
        σ = pm.Exponential('sigma', 1)
        y = pm.Normal('y', μ, σ, observed=obs, shape=x0.shape)
        quap = sts.quap()
    return quap


# Build all three models
m5_1 = build_linear_model(df, x='x_real', y='y')
m5_2 = build_linear_model(df, x='x_spur', y='y')
m5_3 = build_linear_model(df, x='x_real', x1='x_spur', y='y')

print()
print('y ~ x_real:')
sts.precis(m5_1)

print()
print('y ~ x_spur:')
sts.precis(m5_2)

print()
print('y ~ x_real, x_spur:')
sts.precis(m5_3)

# ----------------------------------------------------------------------------- 
#         Plot the data in 3D space to see the fitted plane
# -----------------------------------------------------------------------------
A = x_real
M = x_spur
D = y

Ag, Mg = np.ogrid[-4:4, -4:4]
Dg = m5_3.coef['α'] + m5_3.coef['β'] * Ag + m5_3.coef['β_1'] * Mg

# Plot bi-linear model slices
A_s = M_s = np.arange(-4, 5)
D_M0 = m5_3.coef['α'] + m5_3.coef['β'] * A_s
D_A0 = m5_3.coef['α'] + m5_3.coef['β_1'] * M_s

# Plot univariate model projections
D_A = m5_1.coef['α'] + m5_1.coef['β'] * A_s
D_M = m5_2.coef['α'] + m5_2.coef['β'] * M_s

fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot(projection='3d')

# Plot 3D data and fitted surface
ax.scatter(A, M, D, c='k', alpha=0.4)
ax.plot_surface(Ag, Mg, Dg, edgecolor='k', lw=0.5, alpha=0.2)
ax.plot(A_s, D_M0, zdir='y', color='k')
ax.plot(M_s, D_A0, zdir='x', color='k')

# Plot the projections onto the "walls" of the graph
ax.scatter(A, D, zdir='y', zs=max(M_s), color='C0', alpha=0.4)
ax.plot(A_s, D_A, zdir='y', zs=max(M_s), color='C0')

ax.scatter(M, D, zdir='x', zs=min(A_s), color='C3', alpha=0.4)
ax.plot(M_s, D_M, zdir='x', zs=min(A_s), color='C3')

ax.set(xlabel=r'$x_{\mathrm{real}}$', xlim=(min(A_s), max(A_s)),
       ylabel=r'$x_{\mathrm{spur}}$', ylim=(min(M_s), max(M_s)),
       zlabel=r'$y$',
       proj_type='ortho')

# ax.view_init(0, 0, 0)  # projection onto M-D plane
ax.view_init(0, -90, 0)  # projection onto A-D plane

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
