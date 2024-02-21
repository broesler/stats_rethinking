#!/usr/bin/env python3
# =============================================================================
#     File: cafes.py
#  Created: 2024-02-19 21:04
#   Author: Bernie Roesler
#
"""
§14.1 Varying Slopes at Cafés.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor as pt
import seaborn as sns

from matplotlib import transforms
from matplotlib.patches import Circle
from scipy import linalg, stats

import stats_rethinking as sts


def confidence_ellipse(mean, cov, ax=None,
                       level=0.95, facecolor='none', **kwargs):
    """Plot a confidence ellipse for a 2D Gaussian.

    Parameters
    ----------
    mean : (2,) array_like
        The center of the ellipse.
    cov : (2, 2) array_like
        The covariance matrix. Must be symmetric positive semi-definite.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, the current axes will be used.
    level : float, optional
        The level of the confidence interval to plot. Default is 0.95.
    facecolor : str, optional
        The color with which to fill the ellipse. Default is 'none'.
    **kwargs : dict
        Additional arguments to pass to the Ellipse call.

    Returns
    -------
    circle : matplotlib.patches.Circle
        The circle object.

    References
    ----------
    [0]: <https://gist.github.com/CarstenSchelp/b992645537660bda692f218b562d0712?permalink_comment_id=3465086istcomment-3465086>
    """
    assert mean.shape[0] == 2, "Mean must be length 2"
    assert cov.shape == (2, 2), "Covariance matrix must be 2x2"
    assert np.allclose(cov, cov.T), "Covariance matrix must be symmetric"
    ax = ax or plt.gca()
    kwargs.update(dict(facecolor=facecolor))

    # The scaling parameter is chi-squared with N dof for N-D data
    r = np.sqrt(stats.chi2.ppf(level, df=2))

    # Transform matrix is 3x3 to include translation
    T = np.eye(3)
    T[:2, :2] = linalg.sqrtm(cov)
    transform = transforms.Affine2D(matrix=T).scale(r).translate(*mean)

    circle = Circle(
        xy=(0, 0),
        radius=1,
        transform=transform + ax.transData,
        **kwargs
    )
    return ax.add_patch(circle)


# Simulate the cafes (R code 14.1)
a = 3.5    # avg morning wait time
b = -1     # avg *difference* in afternoon wait time
σ_a = 1    # std in intercepts
σ_b = 0.5  # std in slopes
ρ = -0.7   # correlation between intercepts and slopes

Mu = np.r_[a, b]  # (R code 14.2)

# Directly define covariance matrix (R code 14.3)
cov_ab = σ_a * σ_b * ρ
Σa = np.c_[[σ_a**2, cov_ab],
           [cov_ab, σ_b**2]]

# Define via multiply with correlation matrix (R code 14.5)
sigmas = np.r_[σ_a, σ_b]
Rho = np.c_[[1, ρ],
            [ρ, 1]]
Sigma = np.diag(sigmas) @ Rho @ np.diag(sigmas)

np.testing.assert_allclose(Σa, Sigma)

# (R code 14.6, 14.7)
N_cafes = 20
rng = np.random.default_rng(seed=56)
vary_effects = stats.multivariate_normal.rvs(
    mean=Mu, cov=Sigma, size=N_cafes, random_state=rng
)

# (R code 14.8)
a_cafe, b_cafe = vary_effects.T

qs = np.r_[0.1, 0.3, 0.5, 0.8, 0.99]

# Plot the slopes and intercepts
fig, ax = plt.subplots(num=1, clear=True)

for level in qs:
    confidence_ellipse(Mu, Sigma, ax=ax, level=level, ec='k', alpha=0.4)

ax.scatter(a_cafe, b_cafe, ec='k', fc='none')
ax.set(xlabel='intercepts (a_cafe)',
       ylabel='slopes (b_cafe)')

# Simulate the robot visits (R code 14.10)
rng = np.random.default_rng(seed=22)
N_visits = 10
afternoon = np.tile([0, 1], N_visits*N_cafes // 2)  # indicator for afternoon
cafe_id = np.repeat(range(N_cafes), N_visits)       # unique id
μ = a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon   # the linear model
σ = 0.5                                             # std *within* a cafe

wait = stats.norm.rvs(loc=μ, scale=σ, size=N_visits*N_cafes, random_state=rng)

df = pd.DataFrame(dict(cafe=cafe_id, afternoon=afternoon, wait=wait))

# -----------------------------------------------------------------------------
#         Plot correlation density
# -----------------------------------------------------------------------------
# Draw 10,000 samples from the LKJ distribution with different η
ηs = [1, 2, 4]
# (10_000, 3) for plotting
Rs = np.hstack([pm.draw(pm.LKJCorr.dist(n=2, eta=η), 10_000) for η in ηs])

fig, ax = plt.subplots(num=2, clear=True)
sns.kdeplot(Rs, bw_adjust=0.5, ax=ax)
ax.legend(labels=[f"{η = }" for η in ηs])
ax.set(xlabel='Correlation (ρ)',
       ylabel='Density')


# -----------------------------------------------------------------------------
#         Build the Model (R code 14.12)
# -----------------------------------------------------------------------------
η = 2  # LKJ parameter
with pm.Model():
    A = pm.ConstantData('A', df['afternoon'])
    C = pm.ConstantData('C', df['cafe'])
    a = pm.Normal('a', 5, 2)
    b = pm.Normal('b', -1, 0.5)
    σ_cafe = pm.Exponential('σ_cafe', 1, shape=(2,))
    ρ = pm.LKJCorr('ρ', n=2, eta=η, return_matrix=True)
    # Compute Σ from σ_cafe and R
    S = pt.tensor.eye(2) * σ_cafe
    Σ = pm.math.dot(S, pm.math.dot(ρ, S))
    M = pm.MvNormal('M', mu=pm.math.stack([a, b]), cov=Σ, shape=(N_cafes, 2))
    a_c = pm.Deterministic('a_c', M[:, 0])
    b_c = pm.Deterministic('b_c', M[:, 1])
    σ = pm.Exponential('σ', 1)
    μ = pm.Deterministic('μ', a_c[C] + b_c[C] * A)
    W = pm.Normal('W', μ, σ, shape=μ.shape, observed=df['wait'])
    m14_1 = sts.ulam(data=df, nuts_sampler='numpyro')


# Plot prior and posterior distributions of correlation (R code 14.13)
prior = pm.sample_prior_predictive(model=m14_1.model, samples=10_000)
post = m14_1.get_samples()

fig, ax = plt.subplots(num=3, clear=True)
sns.kdeplot(prior.prior['ρ'].values.flat, c='k', ls='--', ax=ax, label='prior')
sns.kdeplot(post['ρ'].values.flat, c='C0', ax=ax, label='posterior')
ax.legend()
ax.set(xlabel='Correlation',
       ylabel='Density')


# -----------------------------------------------------------------------------
#         Plot "shrinkage" of pooled estimates from unpooled (R code 14.14)
# -----------------------------------------------------------------------------
g = df.groupby(['cafe', 'afternoon'])['wait'].mean()
a_u = g.xs(0, level='afternoon')        # morning wait time in data
b_u = g.xs(1, level='afternoon') - a_u  # difference from morning

sample_dims = ('chain', 'draw')
a_p = m14_1.deterministics['a_c'].mean(sample_dims)
b_p = m14_1.deterministics['b_c'].mean(sample_dims)

# Compute posterior mean Gaussian
post = m14_1.get_samples()
Mu_est = np.r_[post['a'].mean(sample_dims), post['b'].mean(sample_dims)]
rho_est = post['ρ'].mean(sample_dims)
sa_est, sb_est = post['σ_cafe'].mean(sample_dims).T
cov_ab_est = float(sa_est * sb_est * rho_est)
Σ_est = np.c_[[sa_est**2, cov_ab_est],
              [cov_ab_est, sb_est**2]]

# Plot the shrinkage
fig, ax = plt.subplots(num=4, clear=True)
ax.scatter(a_u, b_u, c='C0', label='unpooled')
ax.scatter(a_p, b_p, ec='k', fc='none', label='pooled')

# TODO plot lines between points

ax.set(xlabel='intercept',
       ylabel='slope')

for level in qs:
    confidence_ellipse(Mu_est, Σ_est, ax=ax, level=level, ec='k', alpha=0.4)


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
