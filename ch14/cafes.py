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
from matplotlib.patches import Ellipse
from scipy import stats

import stats_rethinking as sts


# TODO convert from n_std to quantile
def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """Plot an ellipse showing the confidence region of a 2D Gaussian."""
    assert mean.shape == (2,), "Mean must be length 2"
    assert cov.shape == (2, 2), "Covariance matrix must be 2x2"

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this 2D dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=2*ell_radius_x,
                      height=2*ell_radius_y,
                      facecolor=facecolor,
                      **kwargs)

    # Calculate the standard deviation of x from the square root of the
    # variance and multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


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

# TODO test is stats.norm.ppf(level) is the correct contour line
# Grid of Gaussian values to plot contours (R code 14.9)
xg, yg = np.mgrid[1:7:0.01, -2.5:0.5:0.01]
zg = stats.multivariate_normal.pdf(np.dstack((xg, yg)), mean=Mu, cov=Sigma)
zg = (zg.max() - zg) / zg.max()
qs = [0.1, 0.3, 0.5, 0.8, 0.99]

# Plot the slopes and intercepts
fig, ax = plt.subplots(num=1, clear=True)
cs = ax.contour(xg, yg, zg, cmap='Blues', levels=qs)
ax.clabel(cs, qs, inline=True)
for level in qs:
    confidence_ellipse(Mu, Sigma, ax, n_std=stats.norm.ppf(level), ec='k')
ax.scatter(a_cafe, b_cafe, ec='k', fc='none')
ax.set(xlabel='intercepts (a_cafe)',
       ylabel='slopes (b_cafe)')

# Simulate the robot visits (R code 14.10)
N_visits = 10
afternoon = np.tile([0, 1], N_visits*N_cafes // 2)  # indicator for afternoon
cafe_id = np.repeat(range(N_cafes), N_visits)       # unique id
μ = a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon   # the linear model
σ = 0.5                                             # std *within* a cafe

wait = stats.norm.rvs(loc=μ, scale=σ, size=N_visits*N_cafes)

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
    a_cafe = pm.Deterministic('a_cafe', M[:, 0])
    b_cafe = pm.Deterministic('b_cafe', M[:, 1])
    σ = pm.Exponential('σ', 1)
    μ = pm.Deterministic('μ', a_cafe[C] + b_cafe[C] * A)
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
post = m14_1.get_samples()
a_p = m14_1.deterministics['a_cafe'].mean(sample_dims)
b_p = m14_1.deterministics['b_cafe'].mean(sample_dims)

# Compute posterior mean Gaussian
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
ax.set(xlabel='intercept',
       ylabel='slope')

for level in [0.1, 0.3, 0.5, 0.8, 0.99]:
    confidence_ellipse(Mu_est, Σ_est, ax, n_std=stats.norm.ppf(level), ec='k')


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
