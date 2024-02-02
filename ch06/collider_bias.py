#!/usr/bin/env python3
# =============================================================================
#     File: collider_bias.py
#  Created: 2023-05-12 16:49
#   Author: Bernie Roesler
#
"""
Description: §6.3 Collider Bias.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats
from scipy.special import expit  # inverse logit
from tqdm import tqdm

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')


def sim_happiness(N_years=1000, max_age=65, N_births=20, aom=18):
    """Agent-based model of happiness and marriage.

    Parameters
    ----------
    N_years : int
        Number of years to simulate.
    max_age : int
        Maximum age of a perosn in the simulation.
    N_births : int
        The annual birthrate.
    aom : int
        Age of marriage.

    Returns
    -------
    result : pd.DataFrame
        DataFrame containing 'happiness', 'married', and 'age' columns.


    Notes
    -----
    See `R package function <https://github.com/rmcelreath/rethinking/blob/2f01a9c5dac4bc6e9a6f95eec7cae268200a8181/R/sim_happiness.R#L3>`_
    """
    A = np.zeros(0)
    H = np.zeros(0)
    M = np.zeros(0)
    for i in tqdm(range(N_years)):
        A += 1                                          # age everyone
        A = np.append(A, np.ones(N_births))             # newborns
        H = np.append(H, np.linspace(-2, 2, N_births))  # happiness is constant
        M = np.append(M, np.zeros(N_births))            # not yet married
        # Some adults get married!
        idx = (A >= aom) & np.logical_not(M)
        M[idx] = stats.bernoulli(expit(H[idx] - 4)).rvs(np.sum(idx))
        # Others die
        dead = A > max_age
        A = A[~dead]
        H = H[~dead]
        M = M[~dead]

    return pd.DataFrame({'age': A.astype(int),
                         'happiness': H,
                         'married': M.astype(int)})


df = sim_happiness(N_years=1000)
sts.precis(df)

fig = plt.figure(1, clear=True, constrained_layout=True)
fig.set_size_inches((8, 4), forward=True)
ax = fig.add_subplot()
ax.scatter('age', 'happiness', data=df.loc[df['married'] == 0],
           edgecolors='k', facecolors='none', alpha=0.4, label='Unmarried')
ax.scatter('age', 'happiness', data=df.loc[df['married'] == 1],
           c='C0', alpha=0.6, label='Married')
ax.set(xlabel='Age',
       ylabel='Happiness')
ax.set_yticks([-2, -1, 0, 1, 2])
ax.legend(bbox_to_anchor=(0.5, 1.01), loc='lower center', ncol=2)


# Build the model for adults only (R code 6.23)
adults = df[df['age'] > 17].copy()
# Define a variable to range from 0 to 1 so that we can expect the slope of
# happiness to be (2 - (-2)) / 1 = 4.
adults['A'] = (adults['age'] - 18) / (65 - 18)  # [0, 1] from 18 to 65

# Model intercepts with the index variable (R code 6.24)
with pm.Model():
    α = pm.Normal('α', 0, 1, shape=(2,))
    β_A = pm.Normal('β_A', 0, 2)
    μ = pm.Deterministic('μ', α[adults['married']] + β_A * adults['A'])
    σ = pm.Exponential('σ', 1)
    happiness = pm.Normal('happiness', μ, σ,
                          observed=adults['happiness'],
                          shape=(2,))
    m6_9 = sts.quap(data=adults)

print('m6.9:')
sts.precis(m6_9)

# Model happiness *without* controlling for marriage (R code 6.25)
with pm.Model():
    α = pm.Normal('α', 0, 1)
    β_A = pm.Normal('β_A', 0, 2)
    μ = pm.Deterministic('μ', α + β_A * adults['A'])
    σ = pm.Exponential('σ', 1)
    happiness = pm.Normal('happiness', μ, σ, observed=adults['happiness'])
    m6_10 = sts.quap(data=adults)

print('m6.10:')
sts.precis(m6_10)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
