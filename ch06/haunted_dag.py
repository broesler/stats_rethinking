#!/usr/bin/env python3
# =============================================================================
#     File: haunted_dag.py
#  Created: 2023-05-13 00:15
#   Author: Bernie Roesler
#
"""
Description: §6.3.2 The Haunted DAG. Simulate the DAG of influence on
childrens' education:

G -> P -> C
G -> C
* U -> P *
* U -> C *

where G = grandparents, P = parents, C = children, and U is an unknown.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# Define the slopes (R code 6.26)
N = 200   # number of families
b_GP = 1  # direct effect of G on P
b_GC = 0  # direct effect of G on C
b_PC = 1  # direct effect of P on C
b_U = 2   # direct effect of U on P and C

# Draw random observations (R code 6.27)
U = 2 * stats.bernoulli(0.5).rvs(N) - 1         # {-1, 1} U is independent
G = stats.norm.rvs(size=N)                      # G is independent
P = stats.norm(b_GP*G + b_U*U).rvs(N)           # P = f(G, U)
C = stats.norm(b_PC*P + b_GC*G + b_U*U).rvs(N)  # C = f(P, G, U)

df = pd.DataFrame({'C': C, 'P': P, 'G': G, 'U': U})

with pm.Model():
    a = pm.Normal('a', 0, 1)
    b_PC = pm.Normal('b_PC', 0, 1)
    b_GC = pm.Normal('b_GC', 0, 1)
    mu = pm.Deterministic('mu', a + b_PC * df['P'] + b_GC * df['G'])
    sigma = pm.Exponential('sigma', 1)
    C = pm.Normal('C', mu, sigma, observed=df['C'])
    m6_11 = sts.quap(data=df)

print('m6.11:')
sts.precis(m6_11)

# Plot the grandparents' education vs children's (Figure 6.6)
gn = df['U'] == 1  # good neighborhoods

# Highlight parents in the 45th to 60th centiles
pct = np.quantile(df['P'], q=[0.45, 0.60])
mid = (pct[0] <= df['P']) & (df['P'] <= pct[1])

# Regress C on G, conditioning on just these mid parents
with pm.Model():
    a = pm.Normal('a', 0, 1)
    b_GC = pm.Normal('b_GC', 0, 1)
    mu = pm.Deterministic('mu', a + b_GC * df.loc[mid, 'G'])
    sigma = pm.Exponential('sigma', 1)
    C = pm.Normal('C', mu, sigma, observed=df.loc[mid, 'C'])
    mid_quap = sts.quap(data=df)

fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.scatter(df.loc[gn, 'G'], df.loc[gn, 'C'],
           facecolors='none', edgecolors='C0', alpha=0.6)
ax.scatter(df.loc[~gn, 'G'], df.loc[~gn, 'C'],
           facecolors='none', edgecolors='k', alpha=0.6)
ax.scatter(df.loc[gn & mid, 'G'], df.loc[gn & mid, 'C'], c='C0', alpha=0.8,
           label='good neighborhoods')
ax.scatter(df.loc[~gn & mid, 'G'], df.loc[~gn & mid, 'C'], c='k', alpha=0.8,
           label='bad neighborhoods')

# Plot the regression on the middle parents
ax.axline((0, mid_quap.coef['a']), slope=mid_quap.coef['b_GC'], color='k', lw=1)

ax.set(title='Parents in the 45th to 60th centiles',
       xlabel='grandparent education (G)',
       ylabel='child education (C)')
ax.legend()

# Remove the bias by measuring U (R code 6.29)
with pm.Model():
    a = pm.Normal('a', 0, 1)
    b_PC = pm.Normal('b_PC', 0, 1)
    b_GC = pm.Normal('b_GC', 0, 1)
    b_U = pm.Normal('b_U', 0, 1)
    mu = pm.Deterministic('mu', 
                          a + b_PC * df['P'] + b_GC * df['G'] + b_U * df['U'])
    sigma = pm.Exponential('sigma', 1)
    C = pm.Normal('C', mu, sigma, observed=df['C'])
    m6_12 = sts.quap(data=df)

print('m6.12:')
sts.precis(m6_12)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
