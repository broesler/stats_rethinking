#!/usr/bin/env python3
# =============================================================================
#     File: chimpanzees.py
#  Created: 2023-12-05 10:51
#   Author: Bernie Roesler
#
"""
§13.3 Multilevel chimpanzees with social choice. Logistic Regression.
"""
# =============================================================================

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr

from pathlib import Path
from scipy import stats
from scipy.special import expit

import stats_rethinking as sts

# Get the data (R code 11.1), forcing certain dtypes
df = pd.read_csv(
    Path('../data/chimpanzees.csv'),
    dtype=dict({
        'actor': int,
        'condition': bool,
        'prosoc_left': bool,
        'chose_prosoc': bool,
        'pulled_left': bool,
    })
)

df['actor'] -= 1  # python is 0-indexed

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 504 entries, 0 to 503
# Data columns (total 8 columns):
#    Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   actor         504 non-null    category
#  1   recipient     252 non-null    float64
#  2   condition     504 non-null    bool
#  3   block         504 non-null    int64
#  4   trial         504 non-null    int64
#  5   prosoc_left   504 non-null    bool
#  6   chose_prosoc  504 non-null    bool
#  7   pulled_left   504 non-null    bool
# dtypes: bool(4), category(1), float64(1), int64(2)
# memory usage: 14.5 KB

# Define a treatment index variable combining others (R code 11.2)
df['treatment'] = df['prosoc_left'] + 2 * df['condition']  # in range(4)

# -----------------------------------------------------------------------------
#         Create the model
# -----------------------------------------------------------------------------
# m11.4 + cluster for "block" (R code 13.21)
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    block_id = pm.MutableData('block_id', df['block'] - 1)
    # Hyper-priors
    a_bar = pm.Normal('a_bar', 0, 1.5)
    σ_a = pm.Exponential('σ_a', 1)
    σ_g = pm.Exponential('σ_g', 1)
    # Priors
    a = pm.Normal('a', a_bar, σ_a, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    g = pm.Normal('g', 0, σ_g, shape=(len(df['block'].unique()),))      # (6,)
    # Linear model
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(a[actor] + g[block_id] + b[treatment])
    )
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m13_4 = sts.ulam(data=df)

# (R code 13.22)
print('m13.4:')
pt = sts.precis(m13_4)
sts.plot_coef_table(sts.coef_table([m13_4], ['m13.4: block']), fignum=1)

# Plot distributions of deviations by actor and block
fig, ax = plt.subplots(num=2, clear=True)
sns.kdeplot(m13_4.samples['σ_a'].values.flat, bw_adjust=0.5, c='k', label='actor')
sns.kdeplot(m13_4.samples['σ_g'].values.flat, bw_adjust=0.5, c='C0', label='block')
ax.legend()
ax.set(xlabel='Standard Deviation',
       ylabel='Density')


# Model that ignores block, but clusters by actor (R code 13.23)
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    a_bar = pm.Normal('a_bar', 0, 1.5)
    σ_a = pm.Exponential('σ_a', 1)
    a = pm.Normal('a', a_bar, σ_a, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    p = pm.Deterministic('p', pm.math.invlogit(a[actor] + b[treatment]))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m13_5 = sts.ulam(data=df)

# (R code 13.24)
cmp = sts.compare([m13_4, m13_5], mnames=['m13.4: block', 'm13.5: no block'],
                  ic='PSIS', sort=True)

print('ct:')
print(cmp['ct'])

# Model varying effects on the *treatment*  (R code 13.25)
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    block_id = pm.MutableData('block_id', df['block'] - 1)
    # Hyper-priors
    a_bar = pm.Normal('a_bar', 0, 1.5)
    σ_a = pm.Exponential('σ_a', 1)
    σ_b = pm.Exponential('σ_b', 1)
    σ_g = pm.Exponential('σ_g', 1)
    # Priors
    a = pm.Normal('a', a_bar, σ_a, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, σ_b, shape=(len(df['treatment'].unique()),))  # (4,)
    g = pm.Normal('g', 0, σ_g, shape=(len(df['block'].unique()),))      # (6,)
    # Linear model
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(a[actor] + g[block_id] + b[treatment])
    )
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m13_6 = sts.ulam(data=df)

# Compare the two models
ct = sts.coef_table([m13_4, m13_6],
                    ['m13.4 (block)', 'm13.6 (block + treatment)'])
print(ct['coef'].unstack('model').filter(like='b[', axis='rows'))

# -----------------------------------------------------------------------------
#         Handling divergences
# -----------------------------------------------------------------------------
# 1. Try resampling m13.4 for funsies.
# m13_4b = sts.ulam(model=m13_4.model, target_accept=0.99)
# idata = pm.sample(model=m13_4.model, target_accept=0.99)
# print('Divergences: ', int(idata.sample_stats['diverging'].sum()))
# print(f"Acceptance: {float(idata.sample_stats['acceptance_rate'].mean()):.2f}")

# a ~ Normal(a_bar, σ_a) -> z ~ Normal(0, 1), a = a_bar + σ_a*z
# g ~ Normal(    0, σ_g) -> x ~ Normal(0, 1), g =     0 + σ_g*z

# 2. Non-centered model
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    block_id = pm.MutableData('block_id', df['block'] - 1)
    # Hyper-priors
    a_bar = pm.Normal('a_bar', 0, 1.5)
    σ_a = pm.Exponential('σ_a', 1)
    σ_g = pm.Exponential('σ_g', 1)
    # Priors
    z = pm.Normal('z', 0, 1, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    x = pm.Normal('x', 0, 1, shape=(len(df['block'].unique()),))      # (6,)
    # Linear model
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(a_bar + z[actor]*σ_a + x[block_id]*σ_g + b[treatment])
    )
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m13_4nc = sts.ulam(data=df)

# Summary ess_bulk for m13_4 and m13_4nc
neff = pd.DataFrame(dict(
    neff_c=az.summary(m13_4.samples)['ess_bulk'],
    neff_nc=(
        az.summary(m13_4nc.samples)['ess_bulk']
        .rename(lambda x: x.replace('z', 'a'))
        .rename(lambda x: x.replace('x', 'g'))
    )
))
neff['diff'] = neff['neff_nc'] - neff['neff_c']
print(neff)


# -----------------------------------------------------------------------------
#         Posterior Predictions
# -----------------------------------------------------------------------------
# (R code 13.30)
# chimp = 1
# d_pred = dict(
#     actor=np.repeat(chimp, 4),
#     treatment=np.arange(4),
#     block_id=np.repeat(0, 4)
# )
# p_samp = sts.lmeval(m13_4, out=m13_4.model.p, eval_at=d_pred)
# p_mu = p_samp.mean(('chain', 'draw'))
# q = 0.89
# a = (1 - q) / 2
# p_ci = p_samp.quantile([a, 1-a], dim=('chain', 'draw'))

post = m13_4.get_samples()


# Predicting out of sample (R code 3.35)
def p_link_abar(treatment):
    """Model 13.4 link function, ignoring `block`.
    This function sets the actor to the average, ignoring the variation among
    actors."""
    return expit(post['a_bar'] + post['b'].sel(b_dim_0=treatment))


# (R code 13.36)
treatments = np.arange(4)

p_raw = p_link_abar(treatments)
p_mu = p_raw.mean(('chain', 'draw'))
a = (1 - 0.89) / 2
p_ci = p_raw.quantile([a, 1-a], dim=('chain', 'draw'))

# (R code 13.37)
# Simulate some random chimpanzees
a_samp = stats.norm.rvs(loc=post['a_bar'], scale=post['σ_a'])
# Make dimensions compatible for other calcs
a_sim = xr.DataArray(
    a_samp,
    coords=dict(
        chain=range(a_samp.shape[0]),
        draw=range(a_samp.shape[1])
    )
)


def p_link_asim(treatment):
    """Define link function with the simulated chimpanzees.
    This function averages over the uncertainty of the actors."""
    return expit(a_sim + post['b'].sel(b_dim_0=treatment))


p_raw_sim = p_link_asim(treatments)
p_mu_sim = p_raw_sim.mean(('chain', 'draw'))
p_ci_sim = p_raw_sim.quantile([a, 1-a], dim=('chain', 'draw'))


# Plot results
fig, axs = plt.subplots(num=3, ncols=3, sharex=True, sharey=True, clear=True)
ax = axs[0]
ax.set_title('average actor')
ax.plot(treatments, p_mu, 'k-')
ax.fill_between(treatments, p_ci[0], p_ci[1], fc='k', alpha=0.3)

ax = axs[1]
ax.set_title('marginal of actor')
ax.plot(treatments, p_mu_sim, 'k-')
ax.fill_between(treatments, p_ci_sim[0], p_ci_sim[1], fc='k', alpha=0.3)

ax = axs[2]
ax.set_title('simulated actors')
for i in range(100):
    ax.plot(treatments, p_raw_sim.sel(chain=0, draw=i), 'k-', alpha=0.3)

axs[0].set(ylabel='proportion pulled left')
for ax in axs:
    ax.set_xticks(treatments, ['R/N', 'L/N', 'R/P', 'L/P'])
    ax.set(xlabel='treatment',
           xlim=(0, 3),
           ylim=(0, 1))
    ax.spines[['top', 'right']].set_visible(False)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
