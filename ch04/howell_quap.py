#!/usr/bin/env python3
# =============================================================================
#     File: howell_quap.py
#  Created: 2019-07-24 22:32
#   Author: Bernie Roesler
#
"""
  Description: Quadratic approximation to the data (R code 4.26 -- 4.30)
"""
# =============================================================================

import numpy as np
import pandas as pd
import pymc as pm

import stats_rethinking as sts

np.random.seed(56)  # initialize random number generator

# -----------------------------------------------------------------------------
#        Load Dataset
# -----------------------------------------------------------------------------
data_path = '../data/'

# R code 4.26
# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

# Filter adults only
adults = df.loc[df['age'] >= 18]
col = 'height'

# -----------------------------------------------------------------------------
#        Build a quadratic approximation to the posterior
# -----------------------------------------------------------------------------
# Assume: h_i ~ N(mu, sigma)
# Specify the priors for each parameter:
mu_c = 178   # [cm] chosen mean for the height-mean prior
# TODO code both of these into a function.
# mus_c = 20  # [cm] chosen std for the height-mean prior
mus_c = 0.1  # [cm] (R code 4.31) test with narrow prior for the mean
sig_c = 50   # [cm] chosen maximum value for height-stdev prior

Ns = 10_000  # number of samples

# Compute quadratic approximation  (R code 4.27 - 4.28)
with pm.Model() as normal_approx:
    # Define the parameter priors: P(mu), P(sigma)
    mu = pm.Normal('mu', mu=mu_c, sigma=mus_c)
    sigma = pm.Uniform('sigma', 0, sig_c)

    # Define the model likelihood, including the data: P(data | mu, sigma)
    height = pm.Normal('h', mu=mu, sigma=sigma, observed=adults[col])

    # Sample the posterior to find argmax P(mu, sigma | data)  (R code 4.30)
    start = dict(mu=adults[col].mean(),
                 sigma=adults[col].std())

    # normal approximation to the posterior
    quap = sts.quap([mu, sigma], start=start)

print(sts.precis(quap))  # (R code 4.29)
# # Output:
# With mus_c = 20:
#                mean       std        5.5%       94.5%
#   mu     154.607024  0.411994  153.948578  155.265470
#   sigma    7.731333  0.291386    7.265643    8.197024
# With mus_c = 0.1:
#                mean       std        5.5%       94.5%
#   mu     177.863755  0.099708  177.704401  178.023108
#   sigma   24.517564  0.924040   23.040769   25.994359

# Sample from the multivariate posterior (R code 4.34)
#   (other option: use pm.sample() -> trace_to_dataframe())
post = sts.sample_to_dataframe(sts.sample_quap(quap, Ns))

print('covariance:')
print(post.cov())
# covariance:
#              mu     sigma
# mu     0.169549 -0.000837
# sigma -0.000837  0.085236

print('correlation coeff:')
print(post.corr())
# correlation coeff:
#              mu     sigma
# mu     1.000000 -0.006959
# sigma -0.006959  1.000000

# =============================================================================
# =============================================================================
