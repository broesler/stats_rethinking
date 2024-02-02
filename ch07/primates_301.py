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

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# R code 7.33
df = pd.read_csv('../data/Primates301.csv')

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 301 entries, 0 to 300
# Data columns (total 16 columns):
#    Column               Non-Null Count  Dtype
# ---  ------               --------------  -----
#  0   name                 301 non-null    object
#  1   genus                301 non-null    object
#  2   species              301 non-null    object
#  3   subspecies           34 non-null     object
#  4   spp_id               301 non-null    int64
#  5   genus_id             301 non-null    int64
#  6   social_learning      203 non-null    float64
#  7   research_effort      186 non-null    float64
#  8   brain                184 non-null    float64
#  9   body                 238 non-null    float64
#  10  group_size           187 non-null    float64
#  11  gestation            140 non-null    float64
#  12  weaning              116 non-null    float64
#  13  longevity            120 non-null    float64
#  14  sex_maturity         107 non-null    float64
#  15  maternal_investment  104 non-null    float64
# dtypes: float64(10), int64(2), object(4)
# memory usage: 37.8+ KB

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
mnames = ['m7.8 (B and M)', 'm7.9 (B only)', 'm7.10 (M only)']
cmp = sts.compare(models, mnames, sort=True)
ct = cmp['ct']
print(ct)
sts.plot_compare(ct, fignum=1)

# (R code 7.41)
coeftab = sts.coef_table(models, mnames=mnames, params=['bM', 'bB'])
sts.plot_coef_table(coeftab, fignum=2)

# (R code 7.42)
print('corr(log_B, log_M):')
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


# -----------------------------------------------------------------------------
#         Comparing pointwise WAICs
# -----------------------------------------------------------------------------
# (R code 7.43)
tf['waic_m7.8'] = sts.WAIC(m7_8, pointwise=True)['log_L']['WAIC']
tf['waic_m7.9'] = sts.WAIC(m7_9, pointwise=True)['log_L']['WAIC']

# (R code 7.44)
# Scale the points (circle sizes) by the difference of the z-scores of log
# brain volume and log body mass, i.e. large points have large brains for their
# body size.
c = tf['log_B'] - tf['log_M']
c -= min(c)
c /= max(c)
# Convert to real values squared to exaggerate difference in large vs small
s = 50*np.exp(c)**2  # marker area in points**2 == (1/72 in)**2

# Plot it
fig = plt.figure(4, clear=True, constrained_layout=True)
fig.set_size_inches((8, 5), forward=True)
ax = fig.add_subplot()

ax.spines[['right', 'top']].set_visible(False)
ax.axhline(0, c='k', ls='--', lw=1)
ax.axvline(0, c='k', ls='--', lw=1)

tf['waic_diff'] = tf['waic_m7.8'] - tf['waic_m7.9']
ax.scatter(
    x='waic_diff',
    y='log_L',
    data=tf,
    s=s,
    edgecolors='k',
    facecolors='C0',
    alpha=0.4,
    label=r'size = $\left(\frac{B}{M}\right)^2$'
)


# Label the top and bottom 4 x-values.
def annotate_topK(ascending=True, K=5, ax=None):
    """Label the top `K` data points on `ax`."""
    if ax is None:
        ax = plt.gca()
    top_idx = tf.sort_values('waic_diff', ascending=ascending)[-K:].index
    top_names = df.loc[top_idx]['genus'] + '\n' + df.loc[top_idx]['species']
    for i, name in zip(top_idx, top_names):
        ax.text(
            x=tf.loc[i, 'waic_diff'] + 0.02,
            y=tf.loc[i, 'log_L'],
            s=name,
            va='center'
        )


# TODO try adjustText function to reduce overlap?
K = 4
annotate_topK()
annotate_topK(ascending=False)

min_y = tf['log_L'].min()
ax.text(0.02, min_y, 'm7.9 (B only) better →', va='center')
ax.text(-0.02, min_y, '← m7.8 (B and M) better', va='center', ha='right')

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

ax.set(xlabel='pointwise difference in WAIC',
       ylabel='log longevity (std)')


# -----------------------------------------------------------------------------
#         Model brain size as *output* (R code 7.45)
# -----------------------------------------------------------------------------
with pm.Model():
    a = pm.Normal('a', 0, 0.1)
    bM = pm.Normal('bM', 0, 0.5)
    bL = pm.Normal('bL', 0, 0.5)
    μ = pm.Deterministic('μ', a + bM*tf['log_M'] + bL*tf['log_L'])
    σ = pm.Exponential('σ', 1)
    log_B = pm.Normal('log_B', μ, σ, observed=tf['log_B'])
    m7_11 = sts.quap(data=tf)

print('m7.11 (M, L) -> B:')
sts.precis(m7_11, digits=2, verbose=True)


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
