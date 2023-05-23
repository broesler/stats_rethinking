#!/usr/bin/env python3
# =============================================================================
#     File: overfitting.py
#  Created: 2023-05-17 22:42
#   Author: Bernie Roesler
#
"""
§7.1 The Problem with Parameters.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# Manually define data (R code 7.1)
sppnames = ['afarensis', 'africanus', 'habilis', 'boisei', 'rudolfensis',
            'ergaster', 'sapiens']
brainvolcc = [438, 452, 612, 521, 752, 871, 1350]
masskg = [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5]

df = pd.DataFrame(dict(species=sppnames, brain=brainvolcc, mass=masskg))

# Standardize variables (R code 7.2)
df['mass_std'] = sts.standardize(df['mass'])
df['brain_std'] = df['brain'] / df['brain'].max()  # [0, 1]

# Use vague priors (R code 7.3)
with pm.Model():
    α = pm.Normal('α', 0.5, 1)
    β = pm.Normal('β', 0, 10)
    μ = pm.Deterministic('μ', α + β * df['mass_std'])
    log_σ = pm.Normal('log_σ', 0, 1)
    brain_std = pm.Normal('brain_std', μ, pm.math.exp(log_σ),
                          observed=df['brain_std'])
    m7_1 = sts.quap(data=df)

print('m7.1:')
sts.precis(m7_1)

# Compute R² value (R code 7.4)
post = m7_1.sample()
mu_samp = sts.lmeval(m7_1, out=m7_1.model.μ, dist=post)
h_samp = stats.norm(mu_samp, np.exp(post['log_σ'])).rvs()
h_mean = h_samp.mean(axis=1)
r = h_samp.mean(axis=1) - df['brain_std']  # residuals
Rsq = 1 - r.var(ddof=0) / df['brain_std'].var(ddof=0)
print(f"{Rsq = :.4f}")

# NOTE Why is this method incorrect? If we can figure this out, we don't need
# to write the sim function to correspond to link! It is just the lmeval with
# a different output variable.
# ==> I believe this method gives an incorrect answer because it does not touch
# the internal rng state of pytensor for each draw, so the draws are *not*
# independent. Thus, the result is a bunch of random points within the PI of
# brain_std, but whose *means* are not necessarily near the brain_std mean.
#
# h_samp = sts.lmeval(m7_1,
#                     out=m7_1.model.brain_std,
#                     params=[m7_1.model.α,
#                             m7_1.model.β,
#                             m7_1.model.log_σ]
#                     )
#


# Plot the data and linear predictions
fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.scatter('mass', 'brain', data=df)
for label, x, y in zip(df['species'], df['mass'], df['brain']):
    ax.text(x+0.5, y+2, label)
# ax.scatter(df['mass'], h_mean * df['brain'].max(), c='k', marker='x')
ax.set(xlabel='body mass [kg]',
       ylabel='brain volume [cc]')


# -----------------------------------------------------------------------------
#         Create polynomial models (R code 7.6 - 7.8)
# -----------------------------------------------------------------------------
models = dict()
Rsqs = dict()
Np = 6  # max polynomial terms


# (R code 7.5)
def brain_Rsq(quap):
    """Compute the :math:`R^2` value of the model."""
    post = quap.sample()
    mu_samp = sts.lmeval(quap, out=quap.model.μ, dist=post,
                         params=[quap.model.α, quap.model.βn])
    h_samp = stats.norm(mu_samp, np.exp(post['log_σ'])).rvs()
    r = h_samp.mean(axis=1) - df['brain_std']  # residuals
    # pandas default is ddof=1 => N-1, so explicitly use ddof=0 => N.
    return 1 - r.var(ddof=0) / df['brain_std'].var(ddof=0)


# Plot each polynomial fit (Figure 7.3, see plotting_support.R -> brain_plot)
fig = plt.figure(2, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=3, ncols=2)
xe_s = np.linspace(df['mass_std'].min() - 0.2, df['mass_std'].max() + 0.2, 900)

for poly_order in range(1, Np+1):
    with pm.Model():
        ind = pm.MutableData('ind', df['mass_std'])
        α = pm.Normal('α', 0.5, 1, shape=(1,))
        βn = pm.Normal('βn', 0, 10, shape=(poly_order,))
        X = sts.design_matrix(ind.get_value(), poly_order)  # [1, x, x², ...]
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
        log_σ = pm.Normal('log_σ', 0, 1)
        sigma = pm.math.exp(log_σ) if poly_order < 6 else 0.001
        brain_std = pm.Normal('brain_std', μ, sigma, 
                              observed=df['brain_std'], shape=ind.shape)
        # Compute the posterior
        quap = sts.quap(data=df)
        # Store and print the models and R² values
        k = f"m7.{poly_order}"
        models[k] = quap
        Rsqs[k] = brain_Rsq(quap)
        print(k)
        sts.precis(quap)
        print(f"R² = {Rsqs[k]:.4f}")

    # Define the mean
    # TODO fails on creation of design matrix within model.
    # sts.lmplot(quap, mean_var=quap.model.μ, x='mass', y='brain', data=df,
    #            eval_at={'ind': xe}, unstd=True, ax=ax)

    X = sts.design_matrix(xe_s, poly_order)  # (Ne, Np)
    post = quap.sample()
    β = post.drop('log_σ', axis='columns').T  # [α, βn] => (Np, Ns)
    mu_samp = X @ β  # (Ne, Ns)
    mu_mean = mu_samp.mean(axis=1)  # (Ne,)
    mu_pi = sts.percentiles(mu_samp, axis=1)

    xe = sts.unstandardize(xe_s, df['mass'])
    mu_mean = mu_mean * df['brain'].max()
    mu_pi = mu_pi * df['brain'].max()

    # Plot the fit
    i = poly_order - 1
    sharex = ax if i > 0 else None
    ax = fig.add_subplot(gs[i], sharex=sharex)

    ax.scatter('mass', 'brain', data=df)
    ax.plot(xe, mu_mean, 'k')
    ax.fill_between(xe, mu_pi[0], mu_pi[1],
                    facecolor='k', alpha=0.3, interpolate=True)

    ax.set_title(rf"$R^2 = {Rsqs[k]:.2f}$", x=0.02, y=1, loc='left')
    ax.set(xlabel='body mass [kg]',
           ylabel='brain volume [cc]')
    if i < 4:  # all except last row
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=None)
    if i < 4:
        ax.set_ylim((300, 1500))
    elif i == 4:
        ax.set_ylim((0, 2100))
    elif i == 5:
        ax.set_ylim((-500, 2100))

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
