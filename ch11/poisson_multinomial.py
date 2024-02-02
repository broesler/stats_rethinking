#!/usr/bin/env python3
# =============================================================================
#     File: poisson_multinomial.py
#  Created: 2023-12-15 12:00
#   Author: Bernie Roesler
#
"""
§11.2.4 Multinomial as a series of Poissons.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats
from scipy.special import logit, expit

import stats_rethinking as sts

df = pd.read_csv(Path('../data/UCBadmit.csv'))

# binomial model of overall admission probability
with pm.Model():
    a = pm.Normal('a', 0, 100)
    p = pm.Deterministic('p', pm.math.invlogit(a))
    admit = pm.Binomial('admit', df['applications'], p, observed=df['admit'])
    m_binom = sts.ulam(data=df)

# Poisson model of overall admission rate and rejection rate
# TODO fix multiple observed variables issues in Ulam!!
with pm.Model() as model:
    a0 = pm.Normal('a0', 0, 100)
    a1 = pm.Normal('a1', 0, 100)
    λ0 = pm.Deterministic('λ0', pm.math.exp(a0))
    λ1 = pm.Deterministic('λ1', pm.math.exp(a1))
    admit = pm.Poisson('admit', λ0, observed=df['admit'])
    reject = pm.Poisson('reject', λ1, observed=df['reject'])
    m_pois = sts.ulam(data=df)

# Binomial probability of admission
print(f"binomial P(a) = {float(expit(m_binom.coef['a'])):.6f}")

# Poisson probability
k = m_pois.coef
Pp = float(np.exp(k['a0']) / (np.exp(k['a0']) + np.exp(k['a1'])))
print(f"poisson  P(a) = {Pp:.6f}")

# =============================================================================
# =============================================================================
