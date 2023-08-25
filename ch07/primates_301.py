#!/usr/bin/env python3
# =============================================================================
#     File: primates_301.py
#  Created: 2023-08-25 11:30
#   Author: Bernie Roesler
#
"""
§7.5.2 Something about *Cebus*.

Model 301 primate species with the following DAG:

    M -> B -> L
    M -> L
    U -> M
    U -> L

M : Body mass [g]
L : Longevity [maximum lifespan in months]
B : Brain volume [cm³]
U : Unobserved variable
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

# R code 7.33
df = pd.read_csv('../data/Primates301.csv')

# Convert desired variables to log scale (R code 7.34)
df['log_L'] = sts.standardize(np.log(df['longevity']))
df['log_B'] = sts.standardize(np.log(df['brain']))
df['log_M'] = sts.standardize(np.log(df['body']))

# Count missing values (R code 7.35)
print('Missing values:')
print(df[['log_L', 'log_B', 'log_M']].isna().sum())

# Drop the na values (R cod 7.36)
tf = df[['log_L', 'log_B', 'log_M']].dropna()

# Desired model, controlling for M and B -> L (R code 7.37)
with pm.Model():
    a = pm.Normal('a', 0, 0.1)
    bM = pm.Normal('bM', 0, 0.5)
    bB = pm.Normal('bB', 0, 0.5)
    μ = pm.Deterministic('μ', a + bM*tf['log_M'] + bB*tf['log_B'])
    σ = pm.Exponential('σ', 1)
    log_L = pm.Normal('log_L', μ, σ, observed=tf['log_L'])
    m7_8 = sts.quap(data=tf)

# Simple model, controlling for B only (R code 7.38)
with pm.Model():
    a = pm.Normal('a', 0, 0.1)
    bB = pm.Normal('bB', 0, 0.5)
    μ = pm.Deterministic('μ', a + bB*tf['log_B'])
    σ = pm.Exponential('σ', 1)
    log_L = pm.Normal('log_L', μ, σ, observed=tf['log_L'])
    m7_9 = sts.quap(data=tf)

# Simple model, controlling for M only (R code 7.38)
with pm.Model():
    a = pm.Normal('a', 0, 0.1)
    bM = pm.Normal('bM', 0, 0.5)
    μ = pm.Deterministic('μ', a + bM*tf['log_M'])
    σ = pm.Exponential('σ', 1)
    log_L = pm.Normal('log_L', μ, σ, observed=tf['log_L'])
    m7_10 = sts.quap(data=tf)

# Compare the models (R code 7.39-7.40)
models = [m7_8, m7_9, m7_10]
mnames=['m7.8 (B and M)', 'm7.9 (B only)', 'm7.10 (M only)']
cmp = sts.compare(models, mnames, sort=True)
ct = cmp['ct']
print(ct)
sts.plot_compare(ct, fignum=1)

# (R code 7.41)
coeftab = sts.coef_table(models, mnames=mnames, params=['bM', 'bB'])
sts.plot_coef_table(coeftab, fignum=2)

# (R code 7.42)
print('correlation(log_B, log_M):')
print(tf[['log_B', 'log_M']].corr().iloc[0, 1])

# ----------------------------------------------------------------------------- 
#         Figure 7.11
# -----------------------------------------------------------------------------
# Sample the posterior for plotting
post = m7_8.sample(800)

fig = plt.figure(3, clear=True, constrained_layout=True)
fig.set_size_inches((12, 5), forward=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
ax0 = fig.add_subplot(gs[0])  # left side plot
ax1 = fig.add_subplot(gs[1])  # right side plot
ax0.spines[['right', 'top']].set_visible(False)
ax1.spines[['right', 'top']].set_visible(False)

# PLot the data
ax0.scatter('log_M', 'log_B', data=tf, edgecolors='k', facecolors='none')

# Label max/min values
min_i, max_i = tf.sort_values('log_B').index[[0, -1]]
min_name, max_name = df.loc[[min_i, max_i], 'name']
ax0.text(x=tf.loc[min_i, 'log_M'] + 0.075,
         y=tf.loc[min_i, 'log_B'],
         s=min_name)
ax0.text(x=tf.loc[max_i, 'log_M'] - 0.075,
         y=tf.loc[max_i, 'log_B'],
         s=max_name,
         ha='right')

ax0.set(xlabel='log body mass (std)',
        ylabel='log brain volume (std)')

# Plot the posterior distribution of bM vs bB
ax1.scatter('bM', 'bB', data=post, c='k', alpha=0.2)
ax1.set(xlabel='bM',
        ylabel='bB')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
