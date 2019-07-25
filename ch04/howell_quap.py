#!/usr/bin/env python3
#==============================================================================
#     File: howell_quap.py
#  Created: 2019-07-24 22:32
#   Author: Bernie Roesler
#
"""
  Description: Quadratic approximation to the data
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
col = 'height'

#------------------------------------------------------------------------------ 
#        Build a quadratic approximation to the posterior
#------------------------------------------------------------------------------
# Assume: h_i ~ N(mu, sigma)
# Specify the priors for each parameter:
mu_c = 178  # [cm] chosen mean for the height-mean prior
mus_c = 20  # [cm] chosen std  for the height-mean prior
sig_c = 50  # [cm] chosen maximum value for height-stdev prior

# Compute quadratic approximation
with pm.Model() as normal_approx:
    # Define the parameter priors: P(mu), P(sigma)
    mu = pm.Normal('mu', mu=mu_c, sd=mus_c)
    sigma = pm.Uniform('sigma', 0, sig_c)

    # Define the model likelihood, including the data: P(data | mu, sigma)
    height = pm.Normal('h', mu=mu, sd=sigma, observed=adults[col])

    start = dict({'mu': adults[col]

    # Sample the posterior to find P(mu, sigma | data)
    pm.sample()
    map_est = pm.find_MAP()  # use MAP estimation for mean

    # Extract desired values
    mean_mu = map_est['mu']
    std_mu = ((1 / pm.find_hessian(map_est, vars=[mu]))**0.5)[0,0]
    mean_sigma = map_est['sigma']
    std_sigma = ((1 / pm.find_hessian(map_est, vars=[sigma]))**0.5)[0,0]

# quadratic approximation
quap = dict({'mu': stats.norm(mean_mu, std_mu),
             'sigma': stats.norm(mean_sigma, std_sigma)})

print(sts.precis(quap))

#==============================================================================
#==============================================================================
