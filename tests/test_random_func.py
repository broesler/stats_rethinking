#!/usr/bin/env python3
# =============================================================================
#     File: test_random_func.py
#  Created: 2023-06-27 17:49
#   Author: Bernie Roesler
#
"""
Description: Test random number generation in a function.
"""
# =============================================================================

import multiprocessing as mp
import numpy as np

from scipy import stats


def test_rand(N, seed=True):
    """Compute the mean of a sequence of random numbers."""
    # NOTE this line is *required* for expected parallel behavior
    if seed:
        np.random.seed()
    dist = stats.norm(0, 1)
    return dist.rvs(N).mean()


N = 20
Ne = 100


# Parallel:
def test_seed(_):
    return test_rand(N, seed=True)


def test_no_seed(_):
    return test_rand(N, seed=False)


# Parallel
with mp.Pool(16) as pool:
    x_par_seed = pool.map(test_seed, range(Ne))

with mp.Pool(16) as pool:
    x_par_no_seed = pool.map(test_no_seed, range(Ne))

# Non-parallel:
x_non = [test_rand(N) for _ in range(Ne)]

# print(np.c_[sorted(x_par), sorted(x_non)])
print(f"{len(np.unique(x_par_seed)) = }")
print(f"{len(np.unique(x_par_no_seed)) = }")
print(f"{len(np.unique(x_non)) = }")

# =============================================================================
# =============================================================================
