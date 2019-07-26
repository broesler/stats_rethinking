#!/usr/bin/env python3
#==============================================================================
#     File: howell_predictor.py
#  Created: 2019-07-25 21:26
#   Author: Bernie Roesler
#
"""
  Description: Make a linear model of the Howell data.
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

# Filter adults only
adults = df.loc[df['age'] >= 18]

Ns = 10_000

# Plot the raw data
fig = plt.figure(1, clear=True)
ax = fig.add_subplot()
ax.scatter(adults['weight'], adults['height'])
ax.set_xlabel('weight [kg]')
ax.set_ylabel('height [cm]')

#------------------------------------------------------------------------------ 
#        Build a Model
#------------------------------------------------------------------------------
with pm.Model() as linear_model:
    w = adults['weight']                                # independent variable
    alpha = pm.Normal('alpha', mu=178, sd=20)           # parameter priors
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', 0, 50)                  # std prior
    h = pm.Normal('h', mu=alpha + beta*(w - w.mean()),  # likelihood
                  sd=sigma,
                  observed=adults['height'])

    trace = pm.sample(Ns)

df = pm.trace_to_dataframe(trace)

#==============================================================================
#==============================================================================
