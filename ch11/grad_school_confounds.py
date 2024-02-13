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
import pymc as pm
import seaborn as sns

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
    m0 = sts.ulam()

# Model the direct effects, now confounded!
with pm.Model():
    α = pm.Normal('α', 0, 1, shape=(2, 2))
    p = pm.Deterministic('p', pm.math.invlogit(α[G, D]))
    admit = pm.Bernoulli('admit', p, shape=p.shape, observed=A)
    m1 = sts.ulam()

print('mGu:')
sts.precis(m0)

print('mGDu:')
sts.precis(m1)

# Plot the posterior
post = m1.get_samples()
post_p = (
    expit(post['α'])
    .stack(sample=('chain', 'draw'))
    .transpose('sample', ...)
)


def plot_post(post, ax=None):
    """Plot the posterior distribution for each gender and department."""
    if ax is None:
        ax = plt.gca()

    colors = ['C0', 'C3']
    lss = ['-', '--']

    for i, j in itertools.product([0, 1], [0, 1]):
        sns.kdeplot(
            post[:, i, j],
            ls=lss[i],
            color=colors[j],
            label=f"D{j}, G{i}",
            ax=ax
        )

    ax.set_ylabel('Density')
    return ax


fig, axs = plt.subplots(num=1, nrows=2, sharex=True, clear=True)
ax = axs[0]
plot_post(post_p, ax=ax)
ax.legend()
ax.set(title='Ignore Confound')

# -----------------------------------------------------------------------------
#         Sensitivity Analysis
# -----------------------------------------------------------------------------
# Plug in β and γ as "data"
with pm.Model():
    # Declare unobserved variable
    u_sim = pm.Normal('u_sim', 0, 1, shape=(N,))

    # A model
    # High ability has strong effect regardless of gender
    β = pm.MutableData('β', np.r_[1, 1])
    α = pm.Normal('α', 0, 1, shape=(2, 2))
    p = pm.Deterministic('p', pm.math.invlogit(α[G, D] + β[G]*u_sim))
    admit = pm.Bernoulli('admit', p, shape=p.shape, observed=A)

    # D model
    # Only gender 1 is affected by latent ability
    γ = pm.MutableData('γ', np.r_[1, 0])
    δ = pm.Normal('δ', 0, 1, shape=(2,))
    q = pm.Deterministic('q', pm.math.invlogit(δ[G] + γ[G]*u_sim))
    dept = pm.Bernoulli('dept', q, shape=q.shape, observed=D)

    mGDu = sts.ulam()

post_u = (
    expit(mGDu.get_samples()['α'])
    .stack(sample=('chain', 'draw'))
    .transpose('sample', ...)
)

ax = axs[1]
plot_post(post_u, ax=ax)
ax.set(title='Assume Confound',
       xlabel='Probability of Admission')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
