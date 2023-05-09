#!/usr/bin/env python3
# =============================================================================
#     File: set_data_example.py
#  Created: 2023-05-03 15:40
#   Author: Bernie Roesler
#
"""
Description: Test pymc set_data() function.
<https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.set_data.html>
"""
# =============================================================================

# import numpy as np
import pymc as pm

# import pytensor
from pytensor.graph.basic import ancestors, graph_inputs
# from pytensor.tensor.random.var import (
#     RandomGeneratorSharedVariable,
#     RandomStateSharedVariable,
# )
from pytensor.tensor.sharedvar import SharedVariable, TensorSharedVariable
from pytensor.tensor.var import TensorConstant, TensorVariable

import stats_rethinking as sts

# Model with no data constants, just raw inputs
with pm.Model() as model_a:
    beta = pm.Normal('beta', 0, 1)
    mu = pm.Deterministic('mu', [1., 2., 3.] * beta)
    y = pm.Normal('y', mu, 1, observed=[1., 2., 3.])

inputs_a = {v.name: list(graph_inputs([v])) for v in [beta, mu, y]}

# Model with mutable data variables
with pm.Model() as model_b:
    ind = pm.MutableData('ind', [1., 2., 3.])
    obs = pm.MutableData('obs', [1., 2., 3.])
    alpha = pm.Normal('alpha', 0, 1)
    beta = pm.Normal('beta', 0, 1)
    mu = pm.Deterministic('mu', alpha + ind * beta)
    sigma = pm.Normal('sigma', 0, 1)
    y = pm.Normal('y', mu, sigma, observed=obs, shape=ind.shape)

inputs_b = {v.name: list(graph_inputs([v]))
            for v in [ind, obs, alpha, beta, mu, sigma, y]
            }

# [(v, type(v)) for v in inputs_b['mu']]
vinputs_b = [v
             for v in inputs_b['mu']
             if isinstance(v, (TensorVariable, TensorSharedVariable))
                and not isinstance(v, TensorConstant)
             ]

mu_named_inputs = list(sts.named_graph_inputs([mu]))
mu_inputs = sts.inputvars(mu)

# =============================================================================
# =============================================================================
