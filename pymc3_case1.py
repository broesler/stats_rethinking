#!/usr/bin/env python3
#==============================================================================
#     File: pymc3_case1.py
#  Created: 2019-06-20 23:05
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

plt.style.use('seaborn-darkgrid')
np.random.seed(123)  # initialize random number generator

#------------------------------------------------------------------------------ 
#        Case Study #1: Stochastic Volatility
#------------------------------------------------------------------------------
# data_url = 'https://raw.githubusercontent.com/pymc-devs/pymc3/master/pymc3/examples/data/SP500.csv'
data_url = './data/SP500.csv'
returns = pd.read_csv(data_url, parse_dates=True, index_col=0)

fig, ax = plt.subplots(num=1, figsize=(10, 6), clear=True)
returns.plot(ax=ax)
ax.set_ylabel('Daily Returns [%]')

with pm.Model() as sp500_model:
    nu = pm.Exponential('nu', 1/10., testval=5.)
    sigma = pm.Exponential('sigma', 1/0.02, testval=.1)

    s = pm.GaussianRandomWalk('s', sigma=sigma, shape=len(returns))
    volatility_process = pm.Deterministic('volatility_process', pm.math.exp(-2*s)**0.5)

    r = pm.StudentT('r', nu=nu, sigma=volatility_process, observed=returns['change'])

with sp500_model:
    trace = pm.sample(2000)

# Plot MCMC traces for parameter values
pm.traceplot(trace, var_names=['nu', 'sigma'])

# Plot the model vs the actual returns
fig, ax = plt.subplots(num=3, figsize=(15, 8))
returns.plot(ax=ax)
ax.plot(returns.index, 1/np.exp(trace['s',::5].T), 'C3', alpha=.03);
ax.set(title='volatility_process',
       xlabel='time',
       ylabel='volatility')
ax.legend(['S&P500', 'stochastic volatility process']);

plt.show()
#==============================================================================
#==============================================================================
