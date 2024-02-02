#!/usr/bin/env python3
# =============================================================================
#     File: exp_gamma.py
#  Created: 2023-12-18 09:55
#   Author: Bernie Roesler
#
"""
§11.3 Overthinking: Generative models for exponential and gamma distributions.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats


# (R code 11.62-63)
def sim_machine(D, N, M=1, S=100_000):
    """Randomly choose one of the 2 parts to fail on each of 100 days.

    Parameters
    ----------
    D : int
        Number of simulation days.
    N : int
        Number of parts in the machine.
    M : int, optional
        Number of parts that fail.
    S : int, optional
        Number of samples.

    Returns
    -------
    result : (S,) ndarray
        Array of number of days until failure of at least one part.
    """
    return np.sort(stats.uniform(loc=1, scale=D-1).rvs((S, N)), axis=1)[:, M-1]


# Simulate a machine with N parts for D days
D = 100  # simulation days

# Single part failure
x2_1 = sim_machine(D, N=2)
x5_1 = sim_machine(D, N=5)

# Multi-part failure
x10_2 = sim_machine(D, N=10, M=2)
x10_5 = sim_machine(D, N=10, M=5)

# fit an exponential/gamma distribution to the data.
days = np.linspace(0, 100, 1000)
exp_p = stats.expon.fit(x5_1)
expon_fit = stats.expon(*exp_p).pdf(days)

gamma_p2 = stats.gamma.fit(x10_2)
gamma_fit2 = stats.gamma(*gamma_p2).pdf(days)

gamma_p5 = stats.gamma.fit(x10_5)
gamma_fit5 = stats.gamma(*gamma_p5).pdf(days)

norm_p5 = stats.norm.fit(x10_5)
norm_fit5 = stats.norm(*norm_p5).pdf(days)

# Plot the density of the x-values vs the day
fig, axs = plt.subplots(num=1, ncols=2, clear=True)
fig.set_size_inches((12.8, 4.8), forward=True)

sns.kdeplot(x2_1, c='k',  bw_adjust=0.1, ax=axs[0], label='2 parts')
sns.kdeplot(x5_1, c='C0', bw_adjust=0.1, ax=axs[0], label='5 parts')
axs[0].plot(days, expon_fit, 'C0--',
            label=f"Exp({', '.join(f'{x:.2f}' for x in exp_p)})")

axs[0].legend()
axs[0].set(
    title='Single Part Failure: Exponential',
    xlabel='Day',
    ylabel='Density',
)

sns.kdeplot(x10_2, c='k',  bw_adjust=0.1, ax=axs[1], label='2/10 parts')
sns.kdeplot(x10_5, c='C0', bw_adjust=0.1, ax=axs[1], label='5/10 parts')
axs[1].plot(days, gamma_fit2, 'k--',
            label=rf"$\Gamma({', '.join(f'{x:.2f}' for x in gamma_p2)})$")
axs[1].plot(days, gamma_fit5, 'C0--',
            label=rf"$\Gamma({', '.join(f'{x:.2f}' for x in gamma_p5)})$")
axs[1].plot(days, norm_fit5, 'C1--',
            label=rf"$\mathcal{{N}}({', '.join(f'{x:.2f}' for x in norm_p5)})$")

axs[1].legend()
axs[1].set(
    title='Multiple Part Failure: Gamma',
    xlabel='Day',
    ylabel='Density',
)


# =============================================================================
# =============================================================================
