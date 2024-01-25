#!/usr/bin/env python3
# =============================================================================
#     File: 12H1-4.py
#  Created: 2024-01-24 18:50
#   Author: Bernie Roesler
#
"""
Exercises 12H1 through 12H4 (mislabeled in 2ed as 11H1 etc.)
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

df = pd.read_csv(Path('../data/hurricanes.csv'))


# =============================================================================
# =============================================================================
