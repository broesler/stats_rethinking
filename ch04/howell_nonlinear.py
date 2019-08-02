#!/usr/bin/env python3
#==============================================================================
#     File: howell_nonlinear.py
#  Created: 2019-08-01 22:23
#   Author: Bernie Roesler
#
"""
  Description: Build non-linear models of the Howell data
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

from scipy import stats
from matplotlib.gridspec import GridSpec

import stats_rethinking as sts

plt.ion()
plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

#------------------------------------------------------------------------------ 
#        Load Dataset
#------------------------------------------------------------------------------
data_path = '../data/'

# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

Ns = 10_000  # general number of samples to use

# Plot the raw data, separate adults and children
is_adult = df['age'] >= 18
adults = df[is_adult]
children = df[~is_adult]

fig = plt.figure(1, clear=True)
ax = fig.add_subplot()
ax.scatter(adults['weight'], adults['height'], alpha=0.5, label='Adults')
ax.scatter(children['weight'], children['height'], c='C3', alpha=0.5, 
           label='Children')
ax.set(xlabel='weight [kg]',
         ylabel='height [cm]')
ax.legend()

#------------------------------------------------------------------------------ 
#        Build a Polynomial Model of Height
#------------------------------------------------------------------------------
# Standardize the input
w = stats.zscore(df['weight'])  # [-] independent variable

with pm.Model() as poly_model:
    alpha = pm.Normal('alpha', 178, 20)
    beta_0 = pm.Lognormal('beta_0', 0, 1)
    beta_1 = pm.Normal('beta_1', 0, 1)
    sigma = pm.Uniform('sigma', 0, 50)
    mu = alpha + beta_0 * w + beta_1 * w**2
    h = pm.Normal('h', mu=mu, sd=sigma, observed=df['height'])
    quap = sts.quap(dict(alpha=alpha,
                         beta_0=beta_0, 
                         beta_1=beta_1,
                         sigma=sigma))
    tr = sts.sample_quap(quap, Ns)

print(sts.precis(tr))

#==============================================================================
#==============================================================================
