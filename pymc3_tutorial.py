#!/usr/bin/env python3
#==============================================================================
#     File: pymc3_tutorial.py
#  Created: 2019-06-20 11:25
#   Author: Bernie Roesler
#
"""
  Description: A script to execute the code in:
               <https://docs.pymc.io/notebooks/getting_started.html>
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import cm

plt.style.use('seaborn-darkgrid')

np.random.seed(123)  # initialize random number generator

N = 100  # size of dataset

# True parameter values
sigma = 1
beta = np.array([[1, 1, 2.5]]).T  # (p+1,1) include "alpha" intercept in beta

# Predictor variable (N, p+1)
X = np.hstack([np.ones((N,1)),  # add column of ones for intercept
               np.random.randn(N,1),
               np.random.randn(N,1) * 0.2])

p = X.shape[1]-1    # number of features

# Simulate outcome variable (model values + additional noise)
Y = X @ beta + np.random.randn(N,1)*sigma

df = pd.DataFrame(data=np.hstack([X, Y]), columns=['X0', 'X1', 'X2', 'Y'])

# Plot data vs each predictor axis
fig, ax = plt.subplots(2, 1, sharex=True, num=1, figsize=(5,8), clear=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i+1], Y)
    ax[i].set_xlabel(f'$X_{i+1}$')
    ax[i].set_ylabel('$Y$')

# Plot bivariate data
fig = plt.figure(2, clear=True)
ax = fig.add_subplot(111)
cax = ax.scatter(X[:,1], X[:,2], c=Y.squeeze(), cmap=cm.inferno)

cbar = fig.colorbar(cax)
ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')


plt.show()

#==============================================================================
#==============================================================================
