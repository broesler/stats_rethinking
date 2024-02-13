#!/usr/bin/env python3
# =============================================================================
#     File: grad_school.py
#  Created: 2023-12-11 13:23
#   Author: Bernie Roesler
#
"""
§11.1.4 Aggregated binomial: Graduate school admissions.

Lecture 10 addendum on confounds.
The DAG is:
    G -> A
    G -> D -> A
    u -> D
    u -> A

Exceptional (high ability) individuals of gender 1 don't *apply* to Department
1, they apply to Department 2. Therefore, the ability of gender 1 applicants to
Department 1 is lower than those of gender 2, so fewer gender 1 applicants are
accepted to Department 1 even though there is no bias in admissions.
"""
# =============================================================================

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr

from pathlib import Path
from scipy import stats
from scipy.special import expit

import stats_rethinking as sts

# Simulate a confounded applicant pool
N = 2000  # number of applicants

# Even gender distribution
G = np.random.choice(np.r_[0, 1], size=N)

# Sample ability, high = 1, average = 0
u = stats.bernoulli.rvs(p=0.1, size=N)

# Gender 0 tends to apply to department 0, 1 to 1,
# and G=0 with greater ability tend to apply to 1 as well
D = stats.bernoulli.rvs(p=np.where(G == 0, u, 0.75), size=N)

# Acceptance rates -> Department 1 discriminates against gender 0 (first row)
# (2, 2, 2) -> (u, D, G)
p_u = np.stack([
    np.array([[0.1, 0.1],
              [0.1, 0.3]]),
    np.array([[0.3, 0.5],
              [0.3, 0.5]])
])

# Create entire vector for each applicant
p = p_u[u, D, G]

# Simulate acceptance
A = stats.bernoulli.rvs(p=p, size=N)

# Model the total effect of gender
with pm.Model():
    α = pm.Normal('α', 0, 1, shape=(2,))
    p = pm.Deterministic('p', pm.math.invlogit(α[G]))
    admit = pm.Bernoulli('admit', p, shape=p.shape, observed=A)
    mGu = sts.ulam()

# Model the direct effects, now confounded!
with pm.Model():
    α = pm.Normal('α', 0, 1, shape=(2, 2))
    p = pm.Deterministic('p', pm.math.invlogit(α[G, D]))
    admit = pm.Bernoulli('admit', p, shape=p.shape, observed=A)
    mGDu = sts.ulam()

print('mGu:')
sts.precis(mGu)

print('mGDu:')
sts.precis(mGDu)

# Plot the posterior
post = mGDu.get_samples()
post_p = (
    expit(post['α'])
    .stack(sample=('chain', 'draw'))
    .transpose('sample', ...)
)

colors = ['C0', 'C3']
lss = ['-', '--']

fig, ax = plt.subplots(num=1, clear=True)
for i, j in itertools.product([0, 1], [0, 1]):
    sns.kdeplot(
        post_p[:, i, j],
        ls=lss[i],
        color=colors[j],
        label=f"D{j}, G{i}"
    )

ax.legend()
ax.set(xlabel='probability of admission',
       ylabel='Density')


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
