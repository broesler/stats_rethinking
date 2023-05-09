#!/usr/bin/env python3
# =============================================================================
#     File: set_data_example.py
#  Created: 2023-05-03 15:40
#   Author: Bernie Roesler
#
"""
Description: Test pymc set_data() function.
<https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.set_data.html>
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pymc as pm
import xarray as xr

with pm.Model() as model:
    x = pm.MutableData('x', [1., 2., 3.])
    y = pm.MutableData('y', [1., 2., 3.])
    beta = pm.Normal('beta', 0, 1)
    mu = pm.Deterministic('mu', x * beta)
    obs = pm.Normal('obs', mu, 1, observed=y, shape=x.shape)
    map_esta = pm.find_MAP()
    idata = pm.sample()
    ya_samp = pm.sample_posterior_predictive(idata)
    ya = ya_samp.posterior_predictive['obs'].mean(('chain', 'draw'))

plt.close('all')
az.plot_trace(idata)

# Predict at a new set of inputs
with model:
    pm.set_data({'x': [1.5, 5., 6., 9., 12., 15.]})
    # map_estb = pm.find_MAP()  # fails because y not updated
    yb_samp = pm.sample_posterior_predictive(idata)
    yb = yb_samp.posterior_predictive['obs'].mean(('chain', 'draw'))

# xv = xr.DataArray(np.linspace(0, 1, 1000))
# idata.posterior['y_model'] = idata.posterior['beta'] * xv

# az.plot_lm(idata=idata, y='obs', x=xv)

fig, ax = plt.subplots(1, clear=True, constrained_layout=True)
ax.scatter(idata.constant_data['x'], idata.constant_data['y'], 
           c='k', marker='x', label='Observed')
ax.scatter(ya_samp.constant_data['x'], ya, label='Posterior Predictive (a)')
ax.scatter(yb_samp.constant_data['x'], yb, c='C1', label='Posterior Predictive (b)')
ax.plot(yb_samp.constant_data['x'], yb, 'C1')
ax.set(aspect='equal', xlabel='x', ylabel='y')
ax.legend()

# az.plot_ppc(ya_samp, kind='scatter')
# az.plot_lm(idata=idata, y=

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
