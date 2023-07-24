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
from tqdm import tqdm

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

SIGMA_CONST = 0.001

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
h_samp = sts.lmeval(
    m7_1,
    out=m7_1.model.brain_std,
    params=[m7_1.model.α,
            m7_1.model.β,
            m7_1.model.log_σ],
)
h_mean = h_samp.mean(axis=1)
r = h_samp.mean(axis=1) - df['brain_std']  # residuals
Rsq = 1 - r.var(ddof=0) / df['brain_std'].var(ddof=0)
print(f"{Rsq = :.4f}")

# Plot the data and linear predictions
fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.scatter('mass', 'brain', data=df)
for label, x, y in zip(df['species'], df['mass'], df['brain']):
    ax.text(x+0.5, y+2, label)
idx = np.argsort(df['mass'])
ax.plot(df.loc[idx, 'mass'], h_mean[idx] * df['brain'].max(), c='k', marker='x')
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
    params = [quap.model.α]
    var_names = quap.model.named_vars.keys()
    if 'βn' in var_names:
        params.append(quap.model.βn)
    if 'log_σ' in var_names:
        params.append(quap.model.log_σ)
    h_samp = sts.lmeval(quap, out=quap.model.brain_std, params=params)
    r = h_samp.mean(axis=1) - df['brain_std']  # residuals
    # pandas default is ddof=1 => N-1, so explicitly use ddof=0 => N.
    return 1 - r.var(ddof=0) / df['brain_std'].var(ddof=0)


def poly_model(poly_order, x='mass_std', y='brain_std', data=df):
    """Build a polynomial model of the brain ~ mass relationship."""
    with pm.Model():
        ind = pm.MutableData('ind', data[x])
        X = sts.design_matrix(ind, poly_order)  # [1 x x² x³ ...]
        α = pm.Normal('α', 0.5, 1, shape=(1,))
        βn = pm.Normal('βn', 0, 10, shape=(poly_order,))
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
        if poly_order < 6:
            log_σ = pm.Normal('log_σ', 0, 1)
            sigma = pm.math.exp(log_σ)
        else:
            sigma = SIGMA_CONST
        brain_std = pm.Normal('brain_std', μ, sigma,
                              observed=data[y], shape=ind.shape)
        # Compute the posterior
        quap = sts.quap(data=data)
    return quap


# Plot each polynomial fit (Figure 7.3, see plotting_support.R -> brain_plot)
fig = plt.figure(2, clear=True, constrained_layout=True)
fig.set_size_inches((8, 10), forward=True)
gs = fig.add_gridspec(nrows=3, ncols=2)
xe_s = np.linspace(df['mass_std'].min() - 0.2, df['mass_std'].max() + 0.2, 200)

for poly_order in range(1, Np+1):
    # Build the model and fit the posterior
    quap = poly_model(poly_order)

    # Store and print the models and R² values
    k = f"m7.{poly_order}"
    models[k] = quap
    Rsqs[k] = brain_Rsq(quap)
    print(k)
    sts.precis(quap)
    print(f"R² = {Rsqs[k]:.4f}")

    # Plot the fit
    i = poly_order - 1
    sharex = ax if i > 0 else None
    ax = fig.add_subplot(gs[i], sharex=sharex)

    # Sample the posterior manually and explicitly
    mu_samp = sts.lmeval(quap, out=quap.model.μ, eval_at={'ind': xe_s},
                         params=[quap.model.α, quap.model.βn])
    # Re-scale the variables
    mu_samp *= df['brain'].max()
    xe = sts.unstandardize(xe_s, df['mass'])

    # PLot results
    sts.lmplot(fit_x=xe, fit_y=mu_samp,
               x='mass', y='brain', data=df,
               ax=ax,
               line_kws=dict(c='k', lw=1),
               fill_kws=dict(facecolor='k', alpha=0.2))

    ax.set_title(rf"{k}: $R^2 = {Rsqs[k]:.2f}$",
                 x=0.02, y=1, loc='left', pad=-14)
    ax.set(xlabel='body mass [kg]',
           ylabel='brain volume [cc]')

    if i < 4:  # all except last row
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=None)
        ax.set_ylim((300, 1500))
    elif i == 4:
        ax.set_ylim((0, 2100))
    elif i == 5:
        ax.set_ylim((-500, 2100))
        ax.axhline(0, c='k', lw=1, ls='--')


# -----------------------------------------------------------------------------
#         Underfitting (R code 7.11)
# -----------------------------------------------------------------------------
with pm.Model():
    ind = pm.MutableData('ind', df['mass_std'])
    α = pm.Normal('α', 0.5, 1)
    μ = pm.Deterministic('μ', α * pm.math.ones_like(ind))  # NOT function of x!
    log_σ = pm.Normal('log_σ', 0, 1)
    brain_std = pm.Normal('brain_std', μ, pm.math.exp(log_σ),
                          observed=df['brain_std'], shape=ind.shape)
    # Compute the posterior
    m7_7 = sts.quap(data=df)

# Store and print the models and R² values
k = "m7.7"
models[k] = m7_7
Rsqs[k] = brain_Rsq(m7_7)

print(k)
sts.precis(m7_7)
print(f"R² = {Rsqs[k]:.4f}")

# Figure 7.4
fig = plt.figure(3, clear=True, constrained_layout=True)
ax = fig.add_subplot()

# Re-scale the variables
mu_samp = sts.lmeval(m7_7, out=m7_7.model.μ, eval_at={'ind': xe_s},
                     params=[m7_7.model.α])
mu_samp *= df['brain'].max()
xe = sts.unstandardize(xe_s, df['mass'])

# PLot results
sts.lmplot(fit_x=xe, fit_y=mu_samp,
           x='mass', y='brain', data=df,
           ax=ax,
           line_kws=dict(c='k', lw=1),
           fill_kws=dict(facecolor='k', alpha=0.2))

ax.set_title(rf"{k}: $R^2 = {Rsqs[k]:.2f}$", x=0.02, y=1, loc='left', pad=-14)
ax.set(xlabel='body mass [kg]',
       ylabel='brain volume [cc]')


# -----------------------------------------------------------------------------
#         Variance -- Leave-one-out validation
# -----------------------------------------------------------------------------
# Figure 7.5
# Re-fit N=1 and N=4 polynomials to the data, leaving one data point out.

fig = plt.figure(4, clear=True, constrained_layout=True)
fig.set_size_inches((10, 5), forward=True)
gs = fig.add_gridspec(nrows=1, ncols=2)

poly_orders = [1, 4]
for i, poly_order in tqdm(enumerate(poly_orders),
                          total=len(poly_orders),
                          position=0,
                          desc='poly_order'):
    ax = fig.add_subplot(gs[i])
    # Fit to the data - 1 row at a time
    for j in tqdm(range(len(df)),
                  position=1,
                  desc='data pts',
                  leave=False):
        # Create the model
        quap = poly_model(poly_order, x='mass_std', y='brain_std',
                          data=df.drop(j))
        mu_samp = sts.lmeval(quap, out=quap.model.μ, eval_at={'ind': xe_s},
                             params=[quap.model.α, quap.model.βn])
        mu_mean = mu_samp.mean(axis=1) * df['brain'].max()
        ax.plot(xe, mu_mean, 'k', lw=1.5, alpha=0.4)

    # Plot the results together
    ax.scatter('mass', 'brain', data=df)
    if poly_order == 4:
        ax.set_ylim((-200, 2200))
    ax.set_title(f"m7.{poly_order}", x=0.02, y=1, loc='left', pad=-14)
    ax.set(xlabel='body mass [kg]',
           ylabel='brain volume [cc]')

# -----------------------------------------------------------------------------
#         §7.2 Log Pointwise Predictive Density (lppd)
# -----------------------------------------------------------------------------
# Compute the log probabilities of the model (R code 7.14, 7.15)


def lppd(quap, Ns=1000):
    """Compute the log pointwise predictive density for a model."""
    post = quap.sample(Ns)
    mu_samp = sts.lmeval(
            quap,
            out=quap.model.μ,
            dist=post,
            eval_at={'ind': df['mass_std']}
        )
    sigma = np.exp(post['log_σ']) if 'log_σ' in post else SIGMA_CONST
    h_logp = stats.norm(mu_samp, sigma).logpdf(df[['brain_std']])
    return sts.logsumexp(h_logp, axis=1) - np.log(Ns)


# R code 7.14
# R> lppd( m7.1 , n=1e4 )
# [1]  0.6099  0.6483  0.5496  0.6235  0.4648  0.4348 -0.8445
#
# >>> lppd(models['m7.1'])
# === array([ 0.6156,  0.6508,  0.5414,  0.6316,  0.4698,  0.4349, -0.8536])
# OR:
# >>> sts.lppd(m7_1)
# === {'brain_std': array([ 0.6286,  0.6678,  0.5485,  0.6373,  0.4579,  0.4187, -0.8548])}

# R code 7.16
print('lppd:')
print(pd.Series({k: sum(lppd(v)) for k, v in models.items()}))


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
