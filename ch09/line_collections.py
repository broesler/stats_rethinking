#!/usr/bin/env python3
# =============================================================================
#     File: line_collections.py
#  Created: 2023-11-29 18:45
#   Author: Bernie Roesler
#
"""
Test varying line width.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection

x = np.linspace(0, 4*np.pi, 10_000)
y = np.cos(x)

lws = 1 + x[:-1]  # segments between points

pts = np.c_[x, y].reshape(-1, 1, 2)
segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segments, linewidths=lws, color='C0')

fig, ax = plt.subplots(num=1, clear=True)
ax.add_collection(lc)
ax.set(xlim=(0, 4*np.pi), ylim=(-1.1, 1.1))

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
