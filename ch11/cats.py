#!/usr/bin/env python3
# =============================================================================
#     File: cats.py
#  Created: 2024-01-04 17:41
#   Author: Bernie Roesler
#
"""
§11.3.2 Actual Cats

See video lecture: <https://youtu.be/Zi6N3GLUJmw?si=PNoNFyq99ImJUUn7&t=4588>
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

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

# Define the model
with pm.Model():
    days_to_event = pm.MutableData('days_to_event', df['days_to_event'])
    color_id = pm.MutableData('color_id', (df['color'] == 'Black').astype(int))
    adopted = pm.MutableData('adopted', df['out_event'] == 'Adoption')
    α = pm.Normal('α', 0, 1, shape=(2,))  # color_id == 0 or 1
    μ = pm.Deterministic('μ', pm.math.exp(α[color_id]))
    λ = pm.Deterministic('λ', 1/μ)
    # Define probabilities of waiting D days without being adopted
    # D|A = 1 ~ Exponential(λ)       # observed adoptions
    # D|A = 0 ~ Exponential-CCDF(λ)  # not-yet-adopted
    # FIXME off by a factor of ~2 in the output D distribution means?
    # It seems like the model is not actually taking the `not_yet_adopted` into
    # account properly, so we're only getting 1/2 the number of days
    obs_adopted = pm.Exponential('obs_adopted', λ, observed=days_to_event)
    # R: custom(exponential_lccdf(λ):
    # not_yet_adopted = pm.math.exp(-λ * days_to_event)
    # not_yet_adopted = pm.Exponential('nya', λ, observed=days_to_event) / λ
    not_yet_adopted = pm.math.log(obs_adopted / λ)
    D = pm.Deterministic(
        'D',
        pm.math.switch(adopted, obs_adopted, not_yet_adopted),
    )
    m11_14 = sts.ulam(data=df)

sts.precis(m11_14)

# Compute average time to adoption
post = m11_14.get_samples()
post['D'] = np.exp(post['α'])
sts.precis(post)


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
    dens = sts.density(x, adjust=0.5).pdf(x)
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
