#!/usr/bin/env python3
# =============================================================================
#     File: trolley_edu.py
#  Created: 2024-01-22 12:34
#   Author: Bernie Roesler
#
"""
Description:
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy.special import logit, expit

import stats_rethinking as sts

df = pd.read_csv(Path('../data/Trolley.csv'))

# Make a category for education
cats = [
    'Elementary School',
    'Middle School',
    'Some High School',
    'High School Graduate',
    'Some College',
    "Bachelor's Degree",
    "Master's Degree",
    'Graduate Degree',
]

df['edu'] = (
    df['edu']
    .astype('category')
    .cat.reorder_categories(cats, ordered=True)
)

# =============================================================================
# =============================================================================
