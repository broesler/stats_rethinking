#!/usr/bin/env python3
# =============================================================================
#     File: cats.py
#  Created: 2024-01-04 17:41
#   Author: Bernie Roesler
#
r"""
§11.3.2 Actual Cats

The model is the following:

    .. math::
    D_i|A_i = 1 \sim \mathrm{Exponential}(\lambda_i)
    D_i|A_i = 0 \sim \mathrm{Exponential-CCDF}(\lambda_i)
    \lambda_i = \frac{1}{\mu_i}
    \log \mu_i = \alpha_{\mathrm{CID}[i]}

where D is the number of days *without* being adopted, A is an indicator of
adoption, and CID is the color ID: 1 for Black, 0 otherwise.

*See* video lecture: <https://youtu.be/Zi6N3GLUJmw?si=PNoNFyq99ImJUUn7&t=4588>
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

df = pd.read_csv(Path('../data/AustinCats.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 22356 entries, 0 to 22355
# Data columns (total 9 columns):
#    Column         Non-Null Count  Dtype
# ---  ------         --------------  -----
#  0   id             22356 non-null  object
#  1   days_to_event  22356 non-null  int64
#  2   date_out       21807 non-null  object
#  3   out_event      22356 non-null  object
#  4   date_in        22356 non-null  object
#  5   in_event       22356 non-null  object
#  6   breed          22356 non-null  object
#  7   color          22356 non-null  object
#  8   intake_age     22356 non-null  int64
# dtypes: int64(2), object(7)
# memory usage: 1.5 MB

for col in ['date_out', 'date_in']:
    df[col] = pd.to_datetime(df[col])

for col in ['in_event', 'out_event', 'breed', 'color']:
    df[col] = df[col].astype('category')

# Want: probability that cat has *not* yet been adopted
# Estimand: Are black cats less likely to be adopted than other colors?

# See <https://discourse.pymc.io/t/simple-exponential-survival-function/4961>

# FIXME off by a factor of ~2 in the output D distribution means?
# Define the model

# junpenglao's model:
with pm.Model():
    days_to_event = pm.MutableData('days_to_event', df['days_to_event'])
    color_id = pm.MutableData('color_id', (df['color'] == 'Black').astype(int))
    adopted = pm.MutableData('adopted', df['out_event'] == 'Adoption')

    # # Define model parameters with indicator variable (NOT IDEAL)
    # a = pm.Normal('a', 0, 1)
    # b = pm.Normal('b', 0, 1)
    # log_rate = a + b * color_id
    # obs_adopted = pm.Exponential('obs_adopted', pm.math.exp(log_rate), observed=days_to_event)
    # # Correct logp for censored data
    # survival = pm.Potential('survival', pm.math.switch(adopted, 0, -log_rate))

    # Define model parameters (PREFERRED)
    α = pm.Normal('α', 0, 1, shape=(2,))  # color_id == 0 or 1
    μ = pm.math.exp(α[color_id])
    λ = 1/μ
    obs_adopted = pm.Exponential('obs_adopted', λ, observed=days_to_event)
    # Correct logp for censored data
    survival = pm.Potential('survival', pm.math.switch(adopted, 0, α[color_id]))

    m11_14_junpenglao = sts.ulam(data=df)

print('junpenglao:')
sts.precis(m11_14_junpenglao)
post = m11_14_junpenglao.get_samples()
# print(f"D[0] = {np.exp(-post['a']).mean():.2f}")
# print(f"D[1] = {np.exp(-(post['a'] + post['b'])).mean():.2f}")
post['D'] = np.exp(post['α'])
sts.precis(post)

# Test sampling from the posterior predictive
postp_junpenglao = pm.sample_posterior_predictive(
    trace=m11_14_junpenglao.get_samples(),
    model=m11_14_junpenglao.model,
    random_seed=56,
)


# john_c's model
with pm.Model():
    days_to_event = pm.ConstantData('days_to_event', df['days_to_event'])
    color_id = pm.ConstantData('color_id', (df['color'] == 'Black').astype(int))
    adopted = pm.ConstantData('adopted', df['out_event'] == 'Adoption')

    # Define model parameters
    α = pm.Normal('α', 0, 1, shape=(2,))  # color_id == 0 or 1
    μ = pm.math.exp(α[color_id])
    λ = 1/μ

    # obs_adopted = pm.Exponential('obs_adopted', λ, observed=days_to_event)

    # Censored data == not adopted
    # Example directly from:
    # <https://www.pymc.io/projects/docs/en/latest/guides/Probability_Distributions.html#custom-distributions>
    def logp(obs, t, λ):
        r"""The log probability function.

        Corresponds to a probability function:

        .. math::
            p(t|\lambda, \mathrm{obs}) = \prod_i \lambda_i^{\mathrm{obs}_i} e^{\lambda_i t_i}

        Parameters
        ----------
        obs : PyTensor
            The observed variable. obs = 0 if data is "censored", 1 otherwise.
        t : PyTensor
            The length of time (days in this case) until the event.
        λ : PyTensor
            The rate parameter.

        Returns
        -------
        result : PyTensor
            The log probability logp(t | λ, obs).
        """
        return (obs * pm.math.log(λ) - λ * t).sum()

    def random(*dist_params, rng=None, size=None):
        """The function from which to generate random samples.

        Parameters
        ----------
        dist_params : list
            The same parameters as given to `logp`. (t, λ) in this case.
        rng : Generator
            The random number generator used internally.
        size : tuple
            The desired size of the random draw.

        Returns
        -------
        result : (M, N) ndarray
            Matrix of M vectors in K dimensions
        """
        t, λ = dist_params
        return np.where(
            np.array(adopted == 1),
            stats.expon.rvs(scale=1/λ, size=size),                    # not censored = Exp(λ)
            stats.expon.isf(stats.uniform.rvs(size=size), scale=1/λ)  # censored = Exp(λ).isf()
        )

    # NOTE DensityDist -> CustomDist in new API
    exp_surv = pm.CustomDist('exp_surv', days_to_event, λ,
                             logp=logp, observed=adopted,
                             random=random)

    m11_14_john = sts.ulam(data=df)

print('john_c:')
sts.precis(m11_14_john)

# Temp for plots:
m11_14 = m11_14_junpenglao
# sts.precis(m11_14)

# Compute average time to adoption
post = m11_14.get_samples()
post['D'] = np.exp(post['α'])
sts.precis(post)

# TODO make plots of the posterior predictives to compare them
postp_john = pm.sample_posterior_predictive(
    trace=m11_14_john.get_samples(),
    model=m11_14_john.model,
    random_seed=56,
)

# -----------------------------------------------------------------------------
#       Plots
# -----------------------------------------------------------------------------

# Plot the distributions of adoption times for Black cats vs others
fig = plt.figure(1, clear=True)
ax = fig.add_subplot()
for idx, label, c in zip([0, 1], ['Other cats', 'Black cats'], ['C3', 'k']):
    # TODO move to function:
    # sts.plot_density(post['D'].sel(α_dim_0=idx), ax=ax, c=c, label=label)
    x = np.sort(post['D'].sel(α_dim_0=idx).stack(sample=('chain', 'draw')))
    dens = sts.density(x).pdf(x)
    ax.plot(x, dens, c=c, label=label)

ax.legend()
ax.set(title='Distribution of Adoption Times',
       xlabel='waiting time [days]',
       ylabel='density')
ax.spines[['top', 'right']].set_visible(False)


# Plot the probability of not being adopted vs time
d = np.linspace(0, 100)
post['λ'] = 1 / np.exp(post['α'])
λ_samp = (
    post['λ']
    .stack(sample=('chain', 'draw'))
    .expand_dims('d')  # prepare to multiply with `d`
)

fig = plt.figure(2, clear=True)
ax = fig.add_subplot()
for idx, label, c in zip([0, 1], ['Other cats', 'Black cats'], ['C3', 'k']):
    D_samp = np.exp(-λ_samp.sel(α_dim_0=idx) * np.c_[d])  # (d, sample)
    pi = sts.percentiles(D_samp, dim='sample')
    ax.plot(d, D_samp.mean('sample'), c=c, label=label)
    ax.fill_between(d, pi[0], pi[1], facecolor=c, interpolate=True, alpha=0.3)

ax.legend()
ax.set(xlabel='days until adoption',
       ylabel='fraction of cats remaining')
ax.spines[['top', 'right']].set_visible(False)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
