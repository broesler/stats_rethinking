#!/usr/bin/env python3
# =============================================================================
#     File: king_markov.py
#  Created: 2023-11-21 15:35
#   Author: Bernie Roesler
#
"""
A simple Markov chain simulation (R code 9.1).
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(seed=56)

num_weeks = 10_000
positions = np.zeros(num_weeks)
current = 10  # the current island

for i in range(num_weeks):
    # Record current position
    positions[i] = current

    # flip coin to generate proposal of adjacent island
    proposal = current + rng.choice((-1, 1))
    # make sure he loops around the archipelago
    if proposal < 1:
        proposal = 10
    elif proposal > 10:
        proposal = 1

    # Move?
    prob_move = proposal / current
    current = proposal if rng.uniform() < prob_move else current


# Plot results (Figure 9.2)
fig, axs = plt.subplots(num=1, ncols=2, clear=True)
fig.set_size_inches((8, 4), forward=True)
ax0, ax1 = axs

ax0.scatter(
    range(100),
    positions[:100],
    marker='o',
    edgecolor='C0',
    facecolor='none'
)
ax0.set(xlabel='week', ylabel='island')

ax1.hist(positions, bins=0.5 + np.arange(11), rwidth=0.1)
ax1.set(xticks=range(1, 11), xlabel='island', ylabel='number of weeks')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
