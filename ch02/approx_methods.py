#!/usr/bin/env python3
# =============================================================================
#     File: grid_approx.py
#  Created: 2019-06-17 11:17
#   Author: Bernie Roesler
#
"""
  Description: Grid approximation example.
"""
# =============================================================================

# import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(123)  # initialize random number generator

# Possible prior distributions
PRIOR_D = dict({'uniform': {'prior': lambda p: np.ones(p.shape),
                            'title': '$U(0, 1)$'},
                'step': {'prior': lambda p: np.where(p < 0.5, 0, 1),
                         'title': '0 where $p < 0.5$, 1 otherwise'},
                'exp': {'prior': lambda p: np.exp(-5 * np.abs(p - 0.5)),
                        'title': '$-5e^{{|p - 0.5|}}$'}
                })

# -----------------------------------------------------------------------------
#        Define Parameters
# -----------------------------------------------------------------------------
# Data
k = 6  # number of event occurrences, i.e. "heads", or "W" == "water" in book
n = 9  # number of trials, i.e. "tosses", or "W + L" == "water + land" in book

# Grid-search parameters
PRIOR_KEY = 'uniform'  # 'uniform', 'step', 'exp'
Nps = [5, 20]  # range of grid sizes to try

# Compute quadratic approximation (see R code 2.6)
# MAP estimation of the parameter mean
with pm.Model() as normal_approx:
    p = pm.Uniform('p', 0, 1)                   # prior distribution of p
    w = pm.Binomial('w', n=n, p=p, observed=k)  # likelihood

    # Compute quadratic approximation (see R code 2.6)
    globe_qa = sts.quap()
    mean_p, std_p = float(globe_qa.coef['p']), globe_qa.std['p']
    norm_a = stats.norm(mean_p, std_p)
    sts.precis(globe_qa)

    # OR compute using MAP estimation (inside `quap`)
    # map_est = pm.find_MAP()                     # use MAP estimation for mean
    # mean_p = map_est['p']                       # extract desired value

    # # Try code from here:
    # # <https://discourse.pymc.io/t/find-hessian-version-differences/10737/2>
    # # pymc>3 gets Hessian on *transformed* space, which differs from pymc3.
    # # Doesn't work?
    # # p_value = normal_approx.rvs_to_values[p]
    # # p_value.tag.transform = None
    # # p_value.name = p.name

    # See: <https://github.com/pymc-devs/pymc/issues/5443>
    # Remove transform from the variable `p`
    # normal_approx.rvs_to_transforms[p] = None

    # Change name so that we can use `map_est['p']` value
    # p_value = normal_approx.rvs_to_values[p]
    # p_value.name = p.name

    # std_p = ((pm.find_hessian(map_est, vars=[p]))**-0.5)[0, 0]

# # # Instead, recreate the model with `transform=None`
# # with pm.Model() as untransformed_m:
# #     p = pm.Uniform('p', 0, 1, transform=None)
# #     w = pm.Binomial('w', n=n, p=p, observed=k, transform=None)
# #     # The Hessian of a Gaussian == "precision" == 1 / sigma**2
# #     std_p = ((pm.find_hessian(map_est, vars=[p]))**-0.5)[0, 0]

# Calculate percentile interval, assuming normal distribution
prob = 0.89
z = stats.norm.ppf([(1 - prob)/2, (1 + prob)/2])  # Φ^{-1} values
ci = mean_p + std_p * z

print('MAP Estimate')
print('------------')
print('  mean   std  5.5%  94.5%')
print(f"p {mean_p:4.2f}  {std_p:4.2f}  {ci[0]:4.2f}   {ci[1]:4.2f}")

# Normal approximation to the posterior
# norm_a = stats.norm(mean_p, std_p)

# -----------------------------------------------------------------------------
#         MCMC estimation of parameter mean (see R code 2.8)
# -----------------------------------------------------------------------------
Ns = 1000  # number of samples

# Manually do the sampling:
# p_trace = np.empty(Ns)  # initialize array of samples
# p_trace[0] = 0.5
# for i in range(1, Ns):
#     p_new = stats.norm.rvs(loc=p_trace[i-1], scale=0.1)
#     if p_new < 0:
#         p_new = np.abs(p_new)
#     if p_new > 1:
#         p_new = 2 - p_new
#     q0 = stats.binom.pmf(k, n, p_trace[i-1])
#     q1 = stats.binom.pmf(k, n, p_new)
#     t = stats.uniform.rvs()
#     p_trace[i] = p_new if t < q1/q0 else p_trace[i-1]

with normal_approx:
    p_trace = pm.sample(Ns).posterior['p']  # extract relevant values

# Analytical Posterior
Beta = stats.beta(k+1, n-k+1)  # Beta(\alpha = 1, \beta = 1) == U(0, 1)

# -----------------------------------------------------------------------------
#        Plot Results
# -----------------------------------------------------------------------------
# Figure 2.7
fig = plt.figure(1, figsize=(8, 6), clear=True, constrained_layout=True)
ax = fig.add_subplot()

prior_func = PRIOR_D[PRIOR_KEY]['prior']

# Plot grid approximation posteriors (see R code 2.3)
for i, Np in enumerate(Nps):
    # Generate the posterior samples on a grid of parameter values
    p_grid, posterior, prior = sts.grid_binom_posterior(Np, k, n,
                                                        prior_func=prior_func,
                                                        norm_post=True)
    p_max = p_grid[np.argmax(posterior)]
    p_max = p_max.mean() if p_max.size > 1 else p_max.item()

    # Plot the result
    ax.axvline(p_max, c=f"C{i}", ls='--', lw=1)
    ax.plot(p_grid, posterior / posterior.max(),
            marker='o', markerfacecolor='none',
            label=rf"Np = {Np}, $p_{{max}}$ = {p_max:.2f}")

# Define grid on which to plot KDE of the MCMC p samples
p_fine = np.linspace(0, 1, num=1000)

# Plot the normal approximation
norm_ap = norm_a.pdf(p_fine)
p_max = p_fine[norm_ap.argmax()]
ax.plot(p_fine, norm_ap / norm_ap.max(), c='C3',
        label=rf"Quad Approx: $\mathcal{{N}}({mean_p:.2f}, {std_p:.2f})$"
              + f", $p_{{max}}$ = {p_max:.2f}")
ax.axvline(p_fine[norm_ap.argmax()], c='C3', ls='--', lw=1)

# Plot the analytical posterior
Beta_p = Beta.pdf(p_fine)
p_max = p_fine[Beta_p.argmax()]
ax.plot(p_fine, Beta_p / Beta_p.max(), 'k-',
        label=rf"True Posterior: $B({k+1}, {n-k+1})$, $p_{{max}}$ = {p_max:.2f}")
ax.axvline(p_max, c='k', ls='--', lw=1)

# Plot the MCMC approximation
# az.plot_trace(p_trace)

# Manually
kde = sts.density(p_trace.stack(sample=('chain', 'draw')))
kde_p = kde.pdf(p_fine)
p_max = p_fine[kde_p.argmax()]
ax.plot(p_fine, kde_p / kde_p.max(),
        c='C4', label=rf"MCMC Posterior, $p_{{max}}$ = {p_max:.2f}")
ax.axvline(p_max, c='C4', ls='--', lw=1)

# Plot the prior
ax.plot(p_fine, prior_func(p_fine), c=0.4*np.array([1, 1, 1]), label='prior')

# Plot formatting
title = rf"$P \sim $ {PRIOR_D[PRIOR_KEY]['title']} |  trials: {n}, events: {k}"
ax.set_title(title)
ax.set_xlabel(r'probability of water, $p$')
ax.set_ylabel(r'posterior probability of $p$')
ax.grid(True)
ax.legend(loc='upper left')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
