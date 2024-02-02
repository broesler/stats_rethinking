#!/usr/bin/env python3
# =============================================================================
#     File: multicollinearity.py
#  Created: 2023-05-11 21:54
#   Author: Bernie Roesler
#
"""
Description: §6.1 Multicollinearity Examples.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from tqdm import tqdm

from pathlib import Path
from scipy import stats
import statsmodels.api as sm

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# -----------------------------------------------------------------------------
#         6.1.1 Simulated Legs
# -----------------------------------------------------------------------------
# (R code 6.2)
N = 100  # number of individuals
height = stats.norm(10, 2).rvs(N)
leg_prop = stats.uniform(0.4, 0.1).rvs(N)  # leg length as proportion of height
leg_left = leg_prop * height + stats.norm(0, 0.02).rvs(N)  # proportion + error
leg_right = leg_prop * height + stats.norm(0, 0.02).rvs(N)
df = pd.DataFrame(np.c_[height, leg_left, leg_right],
                  columns=['height', 'leg_left', 'leg_right'])

# Model with both legs (R code 6.3)
with pm.Model() as model:
    α = pm.Normal('α', 10, 100)
    βl = pm.Normal('βl', 2, 10)
    βr = pm.Normal('βr', 2, 10)
    μ = pm.Deterministic('μ', α + βl * df['leg_left'] + βr * df['leg_right'])
    σ = pm.Exponential('σ', 1)
    height = pm.Normal('height', μ, σ, observed=df['height'])
    m6_1 = sts.quap(data=df)

print('m6.1:')
sts.precis(m6_1)
sts.plot_coef_table(sts.coef_table([m6_1]), fignum=1)  # (R code 6.4)

# Plot the posterior (R code 6.5)
post = m6_1.sample()

fig = plt.figure(2, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
ax = fig.add_subplot(gs[0])
ax.scatter('βr', 'βl', data=post, alpha=0.1)
ax.set(xlabel=r'$\beta_l$',
       ylabel=r'$\beta_r$',
       aspect='equal')

# Plot the posterior sum of the parameters (R code 6.6)
sum_blbr = post['βl'] + post['βr']
ax = fig.add_subplot(gs[1])
# sts.norm_fit(sum_blbr, ax=ax)
sns.histplot(sum_blbr, kde=True, stat='density', ax=ax)
ax.set(xlabel='sum of βl and βr')

# Just model one leg! (R code 6.7)
with pm.Model() as model:
    α = pm.Normal('α', 10, 100)
    βl = pm.Normal('βl', 2, 10)
    μ = pm.Deterministic('μ', α + βl * df['leg_left'])
    σ = pm.Exponential('σ', 1)
    height = pm.Normal('height', μ, σ, observed=df['height'])
    m6_2 = sts.quap(data=df)

print('m6.2:')
sts.precis(m6_2)

# -----------------------------------------------------------------------------
#        Load Dataset (R code 6.8)
# -----------------------------------------------------------------------------
data_path = Path('../data/')
data_file = Path('milk.csv')

df = pd.read_csv(data_path / data_file)

df['K'] = sts.standardize(df['kcal.per.g'])
df['F'] = sts.standardize(df['perc.fat'])
df['L'] = sts.standardize(df['perc.lactose'])

# sns.pairplot(df, vars=['F', 'L', 'K'])


# Model 2 bivariate regressions of fat and lactose (R code 6.9)
def single_regression(data, x, y):
    """Create a single regression model with fixed parameters."""
    with pm.Model():
        ind = pm.MutableData('ind', data[x])
        α = pm.Normal('α', 0, 0.2)
        β = pm.Normal('β', 0, 0.5)
        μ = pm.Deterministic('μ', α + β * ind)
        σ = pm.Exponential('σ', 1)
        K = pm.Normal('K', μ, σ, observed=data[y], shape=ind.shape)
        return sts.quap(data=data)


m6_3 = single_regression(df, 'F', 'K')
m6_3.rename({'β': 'β_F'})
sts.precis(m6_3)

m6_4 = single_regression(df, 'L', 'K')
m6_4.rename({'β': 'β_L'})
sts.precis(m6_4)

# Model both together (R code 6.10)
with pm.Model():
    α = pm.Normal('α', 0, 0.2)
    βF = pm.Normal('βF', 0, 0.5)
    βL = pm.Normal('βL', 0, 0.5)
    μ = pm.Deterministic('μ', α + βF * df['F'] + βL * df['L'])
    σ = pm.Exponential('σ', 1)
    K = pm.Normal('K', μ, σ, observed=df['K'])
    m6_5 = sts.quap(data=df)

sts.precis(m6_5)
# ==> β values are much lower than individual models!


# Plot the effect of correlation of input variables vs posterior std
# (Figure 6.4, R code 6.23)
def sim_collinearity(r=0.9):
    """Simulate two variables that have correlation `r`, and return the
       covariance of perc.fat."""
    df['x'] = (stats.norm(loc=r * df['perc.fat'],
                          scale=np.sqrt((1 - r**2)*df['perc.fat'].var()))
                    .rvs(len(df))
               )
    X = sm.add_constant(df[['perc.fat', 'x']])
    Y = df['K']
    model = sm.OLS(Y, X)
    res = model.fit()
    return np.sqrt(res.cov_params().loc['perc.fat', 'perc.fat'])


def repeat_sim(N=100, r=0.9):
    stddev = [sim_collinearity(r) for _ in range(N)]
    return np.mean(stddev)


r_seq = np.r_[0:0.99:0.01]
stddev = [repeat_sim(N=100, r=r) for r in tqdm(r_seq)]

fig = plt.figure(4, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.plot(r_seq, stddev)
ax.set(xlabel='Cov(F, x)',
       ylabel='Std of the Posterior of F')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
