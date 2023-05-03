#!/usr/bin/env python3
# =============================================================================
#     File: howell_bsplines.py
#  Created: 2019-08-03 08:48
#   Author: Bernie Roesler
#
"""
  Description: Howell model using B-splines (Section 4.5.2)
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

# from scipy import stats
from datetime import datetime

import stats_rethinking as sts

plt.ion()
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(56)  # initialize random number generator

# -----------------------------------------------------------------------------
#        Load Dataset
# -----------------------------------------------------------------------------
data_path = '../data/'

# df: year, doy (day # of year), temp [C], temp_upper [C], temp_lower [C]
df = pd.read_csv(data_path + 'cherry_blossoms.csv')

sts.precis(df)


# Parse year + day of year into an actual datetime column
def convert_time(row):
    """Convert row with 'year' and 'doy' (day of year) columns to datetime."""
    return (datetime(int(row['year']), 1, 1)
            + pd.Timedelta(days=int(row['doy'] - 1)))


df['datetime'] = df.loc[~df['doy'].isnull()].apply(convert_time, axis=1)

Ns = 10_000  # general number of samples to use

# Plot the temperature vs time
fig = plt.figure(1, clear=True, figsize=(12, 4), constrained_layout=True)
ax = fig.add_subplot()
ax.scatter(df['year'], df['temp'], alpha=0.5, label='Raw Data')
ax.set(xlabel='Year',
       ylabel='Temperature [°C]')
ax.legend()

# -----------------------------------------------------------------------------
#        Build B-Splines (R code 4.73 - 4.78)
# -----------------------------------------------------------------------------
df = df.dropna(subset=['temp'])  # only keep rows with temperature data

x = df['year']  # input variable
y = df['temp']  # output variable
Nk = 15         # number of knots
k = 3           # degree of spline (3 == cubic)

# Evenly space knots along percentiles of input variable
# df['year'] is ~uniform so knots will be as well
knots = sts.quantile(x, q=np.linspace(0, 1, Nk))

# Build the basis functions
B = sts.bspline_basis(t=knots, x=x, k=k)

# Figure 4.12: Plot each basis function
fig = plt.figure(2, clear=True, figsize=(8, 6), constrained_layout=True)
gs = fig.add_gridspec(nrows=3, ncols=1)
ax = fig.add_subplot(gs[0])

for i in range(B.shape[1]):
    ax.plot(x, B[:, i], 'k-', alpha=0.5)

# Mark the knots
ax.plot(knots, 1.05*np.ones_like(knots), 'k+', markersize=10, alpha=0.5)

ax.set(xlabel=None,
       ylabel='Basis Value')
ax.tick_params(axis='x', labelbottom=False)

# ----------------------------------------------------------------------------- 
#         Build a model to calculate the weights of the basis functions
# -----------------------------------------------------------------------------
with pm.Model() as bspline_model:
    alpha = pm.Normal('alpha', 6, 10)
    sigma = pm.Exponential('sigma', 1)
    w = pm.Normal('w', 0, 1, shape=B.shape[1])  # one weight per knot
    mu = pm.Deterministic('mu', alpha + pm.math.dot(B, w))
    T = pm.Normal('T', mu, sigma, observed=y)

    quap = sts.quap()
    post = quap.sample(Ns)

# Extract the weights from the sample
w = post.filter(regex='w+').mean(axis=0)
mu_samp = post['alpha'].values[:, np.newaxis] + B @ w

# Plot basis * weights
ax = fig.add_subplot(gs[1], sharex=ax)

for i in range(B.shape[1]):
    ax.plot(x, w[i]*B[:, i], 'k-', alpha=0.5)

ax.plot(knots, 1.05*np.ones_like(knots), 'k+', markersize=10, alpha=0.5)

ax.set(xlabel=None,
       ylabel='Basis * Weight')
ax.tick_params(axis='x', labelbottom=False)

# Plot the fit over the data
q = 0.89  # CI interval probability
mu_mean = mu_samp.mean(axis=0)
mu_hpdi = sts.hpdi(mu_samp, q=q)

ax = fig.add_subplot(gs[2], sharex=ax)
ax.scatter(x, y, alpha=0.5, label='Data')
ax.plot(x, mu_mean, 'k', label='Model')
ax.fill_between(x, mu_hpdi[:, 0], mu_hpdi[:, 1],
                facecolor='k', alpha=0.3, interpolate=True,
                label=rf"{100*q:g}% CI of $\mu$")
ax.set(xlabel='year',
       ylabel='Temperature [°C]')
ax.legend()

# =============================================================================
# =============================================================================
