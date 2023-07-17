#!/usr/bin/env python3
# =============================================================================
#     File: utils.py
#  Created: 2019-06-24 21:35
#   Author: Bernie Roesler
#
"""
  Description: Utility functions for Statistical Rethinking code.
"""
# =============================================================================

import arviz as az
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import sys
import uuid
import warnings

from contextlib import contextmanager
from copy import deepcopy

from pytensor.graph.basic import ancestors
# from pytensor.tensor.random.var import (
#     RandomGeneratorSharedVariable,
#     RandomStateSharedVariable,
# )
from pytensor.tensor.sharedvar import TensorSharedVariable  # , SharedVariable
from pytensor.tensor.var import TensorConstant, TensorVariable

from scipy import stats, linalg
from scipy.interpolate import BSpline
from scipy.special import logsumexp as _logsumexp
from sparkline import sparkify

from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

# TODO
# * define __all__ for the package to declutter internal variables
# * quantile and HPDI (via az.hdi) return transposes of each other for
#   multi-dimensional inputs. Pick one or the other.
# * HPDI does not currently accept multiple q values, but only because the
#   printing function is broken.
# * `pairs` method to plot pair-wise covariance given Quap (sample, then
#   seaborn.pairplot())


def quantile(data, qs=0.89, width=6, precision=4,
             q_func=np.quantile, verbose=False, **kwargs):
    """Pretty-print the desired quantile values from the data.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    width : int, optional, default=10
        Width of printing field.
    precision : int, optional, default=8
        Number of decimal places to print.
    q_func : callable, optional, default=numpy.quantile
        Function to compute the quantile outputs from the data.
    verbose : bool, optional, default=True
        Print the output quantile percentages names and values.
    **kwargs
        Additional arguments to `q_func`.

    Returns
    -------
    quantile : scalar or ndarray
        The requested quantiles. See documentation for `numpy.quantile`.
        Shape = (Q, M), where Q = q.shape, and M is the shape of the data
        without the given axis. If no axis is given, data is unraveled and
        output shape is Q.

    Examples
    --------
    >>> B = np.random.random((5, 10, 7))
    >>> np.quantile(B, q=[1 - 0.89, 0.89]).shape
    === (2,)
    >>> np.quantile(B, q=[1 - 0.89, 0.89], axis=1).shape
    === (2, 5, 7)

    See Also
    --------
    `numpy.quantile`
    """
    qs = np.asarray(qs)
    quantiles = q_func(data, qs, **kwargs)
    if verbose:
        fstr = f"{width}.{precision}f"
        name_str = ' '.join([f"{100*p:{width-1}g}%" for p in np.atleast_1d(qs)])
        value_str = ' '.join([f"{q:{fstr}}" for q in np.atleast_1d(quantiles)])
        print(f"{name_str}\n{value_str}")
    return quantiles


def percentiles(data, q=0.89, **kwargs):
    r"""Pretty-print the desired percentile values from the data.

    .. note:: A wrapper around `quantile`, where the arguments are forced
        to take the form:
    .. math:: a = \frac{1 - q}{2}
        and called with :math:\mathtt{quantile(data, (a, 1-a))}

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 1, inclusive.
    **kwargs
        See `quantile` for additional options.

    Returns
    -------
    percentiles : ndarray
        The requested high and low quantiles.
        Shape = (2, Q, M), where Q = q.shape, and M is the shape of the given
        axis. If no axis is given, data is unraveled and output shape is Q.
        The low boundary is index 0, and the high boundary is index 1.

    See Also
    --------
    quantile
    """
    a = (1 - (q)) / 2
    quantiles = quantile(data, (a, 1-a), **kwargs)
    return quantiles


# TODO remove width and precision arguments and just take fstr='8.2f', e.g.
# * add axis=1 argument ->
# * allow multiple qs, but print them "nested" like on an x-axis.
def hpdi(data, q=0.89, verbose=False, width=6, precision=4,
         axis=None, **kwargs):
    """Compute highest probability density interval.

    .. note::
        This function calls `sts.quantile` with `pymc.stats.hpd` function.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions
    q : array_like
    verbose : bool
    width : int
    precision : int
    axis : int
        If None, unravel the data.
    kwargs : dict_like

    Returns
    -------
    quantiles : (M, N) ndarray
        Matrix of M vectors in N dimensions

        data is (M, N, P)
        az.hdi(data, hdi_prob=0.89).shape == (P, 2)
        np.array([az.hdi(data, hdi_prob=q) for q in qs]).shape == (Q, P, 2)
        ==> az.hdi "ravels" the array along the first 2 dimensions.

    Examples
    --------
    >>> A = np.random.random((5, 10))
    >>> az.hdi(A, hdi_prob=0.89).shape
    === (10, 2)  # FutureWarning (draw, shape) -> (chain, draw)
    >>> B = np.random.random((5, 10, 7))
    >>> az.hdi(B, hdi_prob=0.89).shape
    === (7, 2)
    >>> C = np.random.random((5, 10, 7, 4))
    >>> az.hdi(C, hdi_prob=0.89).shape
    === (7, 4, 2)


    """
    qs = np.asarray(q)
    data = np.asarray(data)

    if verbose and qs.size > 1:
        verbose = False
        warnings.warn("verbose flag only valid for singleton q.")

    if axis is None:
        A = data.reshape(-1).squeeze()  # (M, N) -> (M * N,)
    else:
        # az.hdi "ravels" the array along the first 2 dimensions, because it
        # assumes that the data is from az.convert_to_data() which returns an
        # xarray with (chain, draw, ...) dimensions.
        # ==> move the desired axis to the front, then add 1 dimension.
        # (M, *N*, P) -> (*N*, M, P) -> (1, *N*, M, P)
        A = np.moveaxis(data, axis, 0)[np.newaxis, :]

    # (1, N, M, P) -> (M, P, 2) -> (Q, M, P, 2)
    if qs.ndim == 0:
        Q = az.hdi(A, hdi_prob=qs, **kwargs)
    else:
        Q = np.stack([az.hdi(A, hdi_prob=q, **kwargs) for q in qs])

    Q = np.moveaxis(Q, -1, 0)  # (Q, M, P, 2) -> (2, Q, M, P)

    if verbose:
        fstr = f"{width}.{precision}f"
        name_str = ' '.join([f"{100*p:{width-2}g}%" for p in np.r_[qs, qs]])
        value_str = ' '.join([f"{q:{fstr}}" for q in np.atleast_1d(Q)])
        print(f"|{name_str}|\n{value_str}")

    return Q


def grid_binom_posterior(Np, k, n, prior_func=None, norm_post=True):
    """Posterior probability assuming a binomial distribution likelihood and
    arbitrary prior.

    Parameters
    ----------
    Np : int
        Number of parameter values to use.
    k : int
        Number of event occurrences observed.
    n : int
        Number of trials performed.
    prior_func : callable, optional, default U(0, 1)
        Function of one parameter describing the prior distribution.
        If prior_func is None, it defaults to a uniform prior
    norm_post : bool, optional, default True
        If True, normalize posterior to a maximum value of 1.

    Returns
    -------
    p_grid : (Np, 1) ndarray
        Vector of parameter values.
    posterior : (Np, 1) ndarray
        Vector of posterior probability values.
    """
    p_grid = np.linspace(0, 1, Np)  # vector of possible parameter values
    # default uniform prior
    prior = np.ones(Np) if prior_func is None else prior_func(p_grid)
    likelihood = stats.binom.pmf(k=k, n=n, p=p_grid)  # binomial distribution
    unstd_post = likelihood * prior                   # unnormalized posterior
    posterior = unstd_post / np.sum(unstd_post) if norm_post else unstd_post
    return p_grid, posterior, prior


def density(data, adjust=1.0, **kwargs):
    """Return the kernel density estimate of the data, consistent with
    R function of the same name.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in K dimensions.
    adjust : float, optional, default=1.0
        Multiplicative factor for the bandwidth.
    **kwargs : optional
        Additional arguments passed to `scipy.stats.gaussian_kde`.

    .. note:: This function overrides the `bw_method` argument. The
      stats_rethinking "dens" (R code 2.9) function calls the following
      R function:
          thed <- density(data, adjust=0.5)
      The default bandwidth in `density` (R docs) is: `bw="nrd0"`, which
      corresponds to 'silverman' in python. `adjust` sets `bandwith *= adjust`.

    Returns
    -------
    kde : kernel density estimate object
        Call kde.pdf(x) to get the actual samples

    """
    kde = stats.gaussian_kde(data, **kwargs)
    kde.set_bandwidth(adjust * kde.silverman_factor())
    return kde


# TODO expand documentation with examples
def expand_grid(**kwargs):
    """Return a DataFrame of points, where the columns are kwargs.

    Notes
    -----
    Compare to `numpy.meshgrid`:
        xx, yy = np.meshgrid(mu_list, sigma_list)  # == (..., index='xy')
    `expand_grid` returns the *transpose* of meshgrid's default xy orientation.
    `expand_grid` matches:
        xx, yy = np.meshgrid(mu_list, sigma_list, index='ij')

    See Also
    --------
    numpy.meshgrid
    """
    return pd.DataFrame(itertools.product(*kwargs.values()),
                        columns=kwargs.keys())


# TODO
#   * expand documentation with examples
#   * remove dependence on input type. pd.DataFrame.from_dict? or kwarg?
#       R version uses a LOT of "setMethod" calls to allow function to work
#       with many different datatypes.
#       See: <https://github.com/rmcelreath/rethinking/blob/master/R/precis.r>
#       pythonic way would be to make objects that contain a precis method.
#
def precis(obj, p=0.89, digits=4, verbose=True, hist=True):
    """Return a `DataFrame` of the mean, standard deviation, and percentile
    interval of the given `rv_frozen` distributions.

    Parameters
    ----------
    quap : array-like, DataFrame, or dict
        The model.
    p : float in [0, 1]
        The percentile of which to compute the interval.
    digits : int
        Number of digits in the printed output if `verbose=True`.
    verbose : bool
        If True, print the output.

    Returns
    -------
    result : DataFrame
        A DataFrame with a row for each variable, and columns for mean,
        standard deviation, and low/high percentiles of the variable.
    """
    if not isinstance(obj, (Quap, pd.DataFrame, np.ndarray)):
        raise TypeError(f"quap of type '{type(quap)}' is unsupported!")

    a = (1-p)/2
    pp = 100*np.array([a, 1-a])  # percentages for printing

    if isinstance(obj, Quap):
        title = None
        # Compute density intervals
        z = stats.norm.ppf(1 - a)
        lo = obj.coef - z * obj.std
        hi = obj.coef + z * obj.std
        df = pd.concat([obj.coef, obj.std, lo, hi], axis=1)
        df.columns = ['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"]
        # if hist:
        #     df['histogram'] = sparklines_from_norm(df['mean'], df['std'])

    # DataFrame of data points
    if isinstance(obj, pd.DataFrame):
        obj = obj.select_dtypes(include=np.number)
        title = f"'DataFrame': {obj.shape[0]:d} obs. of {obj.shape[1]} variables:"
        df = pd.DataFrame()
        df['mean'] = obj.mean()
        df['std'] = obj.std()
        for i in range(2):
            df[f"{pp[i]:g}%"] = obj.apply(lambda x: np.nanpercentile(x, pp[i]))
        if hist:
            df['histogram'] = sparklines_from_dataframe(obj)

    # Numpy array of data points
    if isinstance(obj, np.ndarray):
        title = f"'ndarray': {obj.shape[0]:d} obs. of {obj.shape[1]} variables:"
        # Columns are data, ignore index
        vals = np.vstack([np.nanmean(obj, axis=0),
                          np.nanstd(obj, axis=0),
                          np.nanpercentile(obj, pp[0], axis=0),
                          np.nanpercentile(obj, pp[1], axis=0)]).T
        df = pd.DataFrame(vals,
                          columns=['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"]
                          )
        if hist:
            df['histogram'] = sparklines_from_array(obj)

    if verbose:
        if title is not None:
            print(title)
        # Print the dataframe with requested precision
        with pd.option_context('display.float_format',
                               f"{{:.{digits}f}}".format):
            print(df)

    return df


def sparklines_from_norm(means, stds, width=12):
    """Generate list of sparklines from means and stds."""
    # Create matrix of samples
    assert len(means) == len(stds)
    Nm = len(means)
    Ns = 1000
    samp = stats.norm(np.c_[means], np.c_[stds]).rvs(size=(Nm, Ns))
    sparklines = []
    for s in samp:
        sparklines.append(sparkify(np.histogram(s, bins=width)[0]))
    return sparklines


def sparklines_from_dataframe(df, width=12):
    """Generate list of sparklines from a DataFrame."""
    sparklines = []
    for col in df:
        data = df[col].dropna()
        sparklines.append(sparkify(np.histogram(data, bins=width)[0]))
    return sparklines


def sparklines_from_array(arr, width=12):
    """Generate list of sparklines from an array of data."""
    sparklines = []
    for col in arr.T:
        data = col[np.isfinite(col)]
        sparklines.append(sparkify(np.histogram(data, bins=width)[0]))
    return sparklines


# TODO
# * deviance method = -2*self.loglik
# * AIC method = self.deviance + 2*len(self.coef)
class Quap():
    """The quadratic (*i.e.* Gaussian) approximation of the posterior.

    Attributes
    ----------
    coef : dict
        Dictionary of maximum *a posteriori* (MAP) coefficient values.
    cov : (M, M) DataFrame
        Covariance matrix.
    data : (M, N) array_like
        Matrix of the data used to compute the likelihood.
    map_est : dict
        Maximum *a posteriori* estimates of any Deterministic or Potential
        variables.
    loglik : float
        The log-likelihood of the model parameters.
    model : :class:`pymc.Model`
        The pymc model object used to define the posterior.
    start : dict
        Initial parameter values for the MAP optimization. Defaults to
        `model.initial_point`.
    """
    # TODO make all attributes read-only? quap() call populates struct.
    def __init__(self, /, coef=None, cov=None, data=None, map_est=None,
                 loglik=None, model=None, start=None):
        self.coef = coef
        self.cov = cov
        self.data = data
        self.map_est = map_est
        self.loglik = loglik
        self.model = model
        self.start = start

    def __str__(self):
        with pd.option_context('display.float_format', '{:.4f}'.format):
            # remove "dtype: object" line from the Series repr
            meanstr = repr(self.coef).rsplit('\n', 1)[0]

        # FIXME indentation. inspect.cleandoc() fails because the
        # model.str_repr() is not always aligned left.
        out = f"""Quadratic Approximate Posterior Distribution

Formula:
{self.model.str_repr()}

Posterior Means:
{meanstr}

Log-likelihood: {self.loglik:.2f}
"""
        return out

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__str__()}>"

    # TODO use frame_to_dataset to convert this to a dataset by default?
    def sample(self, N=10_000):
        """Sample the posterior approximation.

        Analagous to `rethinking::extract.samples`.
        """
        posterior = stats.multivariate_normal(mean=self.coef, cov=self.cov)
        return pd.DataFrame(posterior.rvs(N), columns=self.coef.index)

    # TODO
    # * rename the model variable itself
    #   now:
    #   >>> model.named_vars
    #   === {'x': x, 'y': y, 'z': z}
    #   >>> model.named_vars['x'].name = 'new_name'
    #   >>> model.named_vars
    #   === {'x': new_name, 'y': y, 'z': z}
    #   Want the key 'x' to be changed to 'new_name' as well.
    # * rename any vector parameters 'b__0', 'b__1', etc.
    def rename(self, mapper):
        """Rename a parameter.

        .. note:: Does NOT work on vector parameters, e.g., 'b__0'.
        """
        self.coef = self.coef.rename(mapper)
        self.cov = self.cov.rename(index=mapper, columns=mapper)
        self.std = self.std.rename(mapper)
        for k, v in mapper.items():
            self.model.named_vars[k].name = v
        return self


def quap(vars=None, var_names=None, model=None, data=None, start=None):
    """Compute the quadratic approximation for the MAP estimate.

    Parameters
    ----------
    vars : list, optional, default=model.unobserved_RVs
        List of variables to optimize and set to optimum
    var_names : list, optional
        List of `str` of variables names specified by `model`
    model : pymc.Model (optional if in `with` context)
    start : `dict` of parameter values, optional, default=`model.initial_point`

    Returns
    -------
    result : dict
        Dictionary of `scipy.stats.rv_frozen` distributions corresponding to
        the MAP estimates of `vars`.
    """
    model = pm.modelcontext(model)

    if vars is None:
        if var_names is None:
            # filter out internally used variables
            mvars, var_names = zip(*[(x, x.name) for x in model.unobserved_RVs
                                     if not x.name.endswith('__')])
        else:
            mvars = [model[x] for x in var_names]
    else:
        if var_names is not None:
            warnings.warn("`var_names` and `vars` set, ignoring `var_names`.")
        mvars = vars
        var_names = [x.name for x in mvars]

    with warnings.catch_warnings():
        # Ignore these warnings:
        # "UserWarning: Intermediate variables (such as Deterministic or
        # Potential) were passed. find_MAP will optimize the underlying
        # free_RVs instead."
        warnings.simplefilter('ignore', category=UserWarning)
        map_est, opt = pm.find_MAP(
                start=start,
                vars=mvars,
                return_raw=True,
                progressbar=False,
                model=model
            )

    # Need to compute *untransformed* Hessian! See ch02/quad_approx.py
    # See: <https://github.com/pymc-devs/pymc/issues/5443>
    for v in mvars:
        try:
            # Remove transform from the variable `v`
            model.rvs_to_transforms[v] = None
            # Change name so that we can use `map_est['v']` value
            v_value = model.rvs_to_values[v]
            v_value.name = v.name
        except KeyError:
            # warnings.warn(f"Hessian for '{v.name}' may be incorrect!")
            continue

    # Build output structure
    # TODO assign in a single call so attributes can be read-only.
    quap = Quap()

    # Filter variables for output
    basic_vars = model.free_RVs
    dnames = [x.name for x in model.deterministics]

    # If requested variables are not basic, just return all of them
    out_vars = set(mvars).intersection(set(basic_vars))
    if not out_vars:
        out_vars = basic_vars

    cnames = []
    hnames = []
    cvals = []
    for ov in out_vars:
        v = ov.name
        x = map_est[v]
        if x.size == 1:
            cnames.append(v)
            cvals.append(float(x))
            hnames.append(v)
        elif x.size > 1:
            # TODO refactor little function to create column names
            # TODO case of 2D, etc. variables
            # Flatten vectors into singletons 'b__0', 'b__1', ..., 'b__n'
            fmt = '02d' if x.size > 10 else 'd'
            cnames.extend([f"{v}__{k:{fmt}}" for k in range(len(x))])
            cvals.extend(x)
            hnames.append(v)  # need the root name for Hessian

    # TODO store coefficients as an xarray Dataset to accomodate
    # multi-dimensional parameters? Having flat Series/DataFrame for means and
    # covariance matrix makes using stats.multivariate_normal simple, but then
    # user-code needs to do the "unflattening" to combine [alpha, b0, b1, ...]
    # for mathematical computations.

    # Coefficients are just the basic RVs, without the observed RVs
    quap.coef = pd.Series({x: v for x, v in zip(cnames, cvals)}).sort_index()
    # The Hessian of a Gaussian == "precision" == 1 / sigma**2
    H = pm.find_hessian(map_est, vars=[model[x] for x in hnames], model=model)
    quap.cov = (pd.DataFrame(linalg.inv(H), index=cnames, columns=cnames)
                  .sort_index(axis=0).sort_index(axis=1))
    quap.std = pd.Series(np.sqrt(np.diag(quap.cov)), index=cnames).sort_index()
    quap.map_est = {k: map_est[k] for k in dnames}
    quap.loglik = opt.fun  # equivalent of sum(loglik(model, pointwise=False))
    quap.model = deepcopy(model)
    quap.start = model.initial_point if start is None else start
    quap.data = deepcopy(data)  # TODO pass data for each call of quap!!
    return quap


# TODO
# * `rethinking::sim` equivalent:
#   - For each Observed variable,
#       - get its inputvars
#       - remove Determinstic vars from inputvars
#       - compile_fn and apply with the posterior samples
#
# * Loop over each Deterministic and Observed variables if none specified,
#   return a dict. Then, lmplot can plot both the mean and the observed
#   confidence intervals in one shot.
# * (un)flatten list of vector or matrix parameters
#   See: the_model.eval_rv_shapes()
# * keep_data=True? Need to get model data for each `eval_at.keys()`
# * refactor `out` to `mean_var` or `lm_var`?
#
def lmeval(fit, out, params=None, eval_at=None, dist=None, N=1000):
    """Sample the indermediate linear models from the given parameter fit.

    .. note:: This function is similar to `rethinking::link`, but with a more
        readable name.

    Parameters
    ----------
    fit : :class:`stats_rethinking.Quap` or similar
        An object containing a pymc model and a posterior distribution.
    out : TensorVariable
        The output variable corresponding to the linear model.
    params : list of TensorVariables
        The parameters of the linear model. If not all parameters are
        specified, the values will be determined by the state of the random
        number generator in the PyTensor graph.
    eval_at : dict like {str: array_like}
        A dictionary of the independent variables over which to evaluate the
        model. Keys must be strings of variable names, and values must be
        array-like, with dimension equivalent to the corresponding variable
        definition in the model.
    dist : dict or DataFrame
        A dict or DataFrame containing samples of the distribution of the
        `params` as values/columns.
    N : int
        If `dist` is None, number of samples to take from `dist`

    Returns
    -------
    samples : (M, N) ndarray
        An array of values of the linear model evaluated at each of M `eval_at`
        points and `N` parameter samples.
    """
    # if out not in fit.model.deterministics:
    #     raise ValueError(f"Variable '{out}' does not exist in the model!")

    if params is None:
        params = inputvars(out)

    if dist is None:
        dist = fit.sample(N)  # take the posterior

    if eval_at is not None:
        pm.set_data(eval_at, model=fit.model)

    if isinstance(dist, pd.DataFrame):
        # Concatenate vector parameters
        dist_t = dict()
        shapes = fit.model.eval_rv_shapes()
        # TODO drop from params if shape == 0?
        for name, s in shapes.items():
            if len(s) > 0 and s[0] > 1:
                dist_t[name] = dist.filter(regex=f"{name}__[0-9]+")
            else:
                dist_t[name] = dist[name]
    else:
        raise TypeError("dist must be a DataFrame!")

    # TODO Update to Python 3.11? PyMC/PyTensor 5.5.0 accepts vector inputs.
    # Compile the graph function to compute. Better than `eval`, which
    # generates a new random state for each call.
    out_func = fit.model.compile_fn(
        inputs=params,
        outs=out,
        on_unused_input='ignore',
    )

    # Manual loop since params are 0-D variables in the model.
    cols = []
    for i in range(len(dist)):
        # Ensure shape of given values matches that of the model variable
        param_vals = {v.name: np.reshape(dist_t[v.name].iloc[i], shapes[v.name])
                      for v in params}
        cols.append(out_func(param_vals))

    return np.array(cols).T  # params as columns


# TODO
# * add "ci" = {'hpdi', 'pi', None} option
# * add option for observed variable and plots its PI too.
def lmplot(quap=None, mean_var=None, fit_x=None, fit_y=None,
           x=None, y=None, data=None,
           eval_at=None, unstd=False, q=0.89, ax=None,
           line_kws=None, fill_kws=None):
    """Plot the linear model defined by `quap`.

    Parameters
    ----------
    fit_x : (N,) array-like
        1-D array of values over which to plot the model fit.
    fit_y : (N, S) array-like
        A 2-D array of `S` samples corresponding to the `N` `fit_x` points.
        .. note:: Either `fit_x` and `fit_y`, or `quap` and `mean_var` must be
            provided to plot the fitted model.
    quap : stats_rethinking.Quap
        The quadratic approximation model estimate.
    mean_var : TensorVariable
        The variable corresponding to the linear model of the mean.
    data : DataFrame
        The data used to fit the model.
    x, y : str
        The column names of the data points to plot.
    eval_at : dict of str: array_like
        A dictionary of the independent variables over which to evaluate the
        model. Keys must be strings of variable names, and values must be
        array-like, with dimension equivalent to the corresponding variable
        definition in the model.
    unstd : bool
        If True, the model was fit to standardized values, so un-standardize
        them to plot in coordinates with real units.
    q : float in [0, 1]
        Quartile over which to shade the mean.
    ax : plt.Axes
        Axes object in which to draw the plot.
    line_kws, fill_kws : dict
        Maps of additional arguments for the line or fill plots.

    Returns
    -------
    ax : plt.Axes
        The axes in which the plot was drawn.
    """
    if ((fit_x is None and fit_y is None)
            and (quap is None and mean_var is None)):
        raise ValueError('Must provide either fit_[xy] or (quap, mean_var)!')

    if ax is None:
        ax = plt.gca()

    if line_kws is None:
        line_kws = dict()
    if fill_kws is None:
        fill_kws = dict()

    if quap is not None and mean_var is not None:
        # TODO? remove ALL of this nonsense and just require mu_samp as input.
        # User code then calls:
        #   mu_s = lmeval()
        #   lmplot(mu_s, ...)
        data_vars = set(named_graph_inputs([mean_var])) - set(inputvars(mean_var))
        data_names = [v.name for v in data_vars]

        if eval_at is None:
            # Use the given data to evaluate the model
            xe = data[x].sort_values()
            if unstd:
                xe = standardize(xe)

            # Determine which name to use
            if len(data_vars) == 1:
                eval_at = {data_names[0]: xe}
            elif x in data_names:
                eval_at = {x: xe}
            elif len(data_vars) > 1:
                raise ValueError("More than 1 data variable in the model!"
                                 + " Please specify `eval_at`")
            # FIXME What happens when "data_vars" is empty?
            # See waic_example.py.
        else:
            if len(eval_at) == 1:
                xe = list(eval_at.values())[0]
            elif x in eval_at:
                xe = eval_at[x]
            else:
                raise ValueError(f"Variable '{x}' not found in model.")

        # Ensure the passed-in variable names match those in the model.
        # If the user has not done quap.rename(), this step should be a no-op.
        name_map = {v.name: k for k, v in quap.model.named_vars.items()}
        eval_rename = dict()
        for k, v in eval_at.items():
            eval_rename[name_map[k]] = v
        eval_at = eval_rename

        mu_samp = lmeval(quap, out=mean_var, eval_at=eval_at)

    elif fit_x is not None and fit_y is not None:
        xe = fit_x
        mu_samp = fit_y

    # Compute mean and error
    mu_mean = mu_samp.mean(axis=1)
    mu_pi = percentiles(mu_samp, q=q, axis=1)  # 0.89 default

    if unstd:
        xe = unstandardize(xe, data[x])
        mu_mean = unstandardize(mu_mean, data[y])
        mu_pi = unstandardize(mu_pi, data[y])

    # Make the plot
    if data is not None:
        ax.scatter(x, y, data=data, alpha=0.4)
    ax.plot(xe, mu_mean, label='MAP Prediction',
            c=line_kws.pop('color', line_kws.pop('c', 'C0')), **line_kws)
    ax.fill_between(xe, mu_pi[0], mu_pi[1],
                    facecolor=fill_kws.pop('facecolor', 'C0'),
                    alpha=fill_kws.pop('alpha', 0.3),
                    interpolate=True,
                    label=rf"{100*q:g}% Percentile Interval of $\mu$",
                    **fill_kws)
    ax.set(xlabel=x, ylabel=y)
    return ax


# -----------------------------------------------------------------------------
#         Graph Utilities
# -----------------------------------------------------------------------------
def named_graph_inputs(graphs, blockers=None):
    """Return list of inputs to a PyTensor variable.

    Parameters
    ----------
    graphs : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.
    blockers : list of `Variable` instances
        A collection of `variable`s that, when found, prevent the graph search
        from preceding from that point.

    Yields
    ------
        Input nodes with a name, in the order found by a left-recursive
        depth-first search started at the nodes in `graphs`.

    See Also
    --------
    pytensor.graph.basic.graph_inputs
    <https://github.com/pymc-devs/pytensor/blob/095e3c9b05525583d3c5a98f9bb75eb6f7ca4556/pytensor/graph/basic.pyL882-L902>
    """
    graphs_names = [v.name for v in graphs]
    yield from (r
                for r in ancestors(graphs, blockers)
                if r.name is not None and r.name not in graphs_names
                )


def inputvars(a):
    """Get the inputs to PyTensor variables.

    Parameters
    ----------
        a: PyTensor variable

    Returns
    -------
        r: list of tensor variables that are inputs

    See Also
    --------
    pymc.inputvars
    <https://www.pymc.io/projects/docs/en/stable/_modules/pymc/pytensorf.html#inputvars>
    """
    return [
        v
        for v in named_graph_inputs(_makeiter(a))
        if isinstance(v, TensorVariable) and not isinstance(v, TensorConstant)
    ]


def _makeiter(a):
    """Return an iterable of the input."""
    return a if isinstance(a, (tuple, list)) else [a]


def norm_fit(data, hist_kws=None, ax=None):
    """Plot a histogram and a normal curve fit to the data."""
    if ax is None:
        ax = plt.gca()
    if hist_kws is None:
        hist_kws = dict()
    sns.histplot(data, stat='density', alpha=0.4, ax=ax, **hist_kws)
    norm = stats.norm(data.mean(), data.std())
    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 1000)
    y = norm.pdf(x)
    ax.plot(x, y, 'C0')
    return ax


def standardize(x):
    """Standardize the input vector `x` by the mean and std of `data`.

    .. note::
        The following lines are equivalent:
                           (x - x.mean()) / x.std() == stats.zscore(x, ddof=1)
        (N / (N-1))**0.5 * (x - x.mean()) / x.std() == stats.zscore(x, ddof=0)
        where N = x.size
    """
    center = x.mean()
    scale = x.std()
    z = (x - center) / scale
    if hasattr(z, 'attrs'):
        z.attrs = {'center': center, 'scale': scale}
    return z


def unstandardize(x, data=None):
    """Return the data to the original scale.

    Parameters
    ----------
    x : array_like
        The scaled data.
    data : array_like
        The un-scaled data with which to compute the center and scale.

    Returns
    -------
    result : un-scaled array of the same size as `x`.
    """
    if data is None:
        try:
            center = x.attrs['center']
            scale = x.attrs['scale']
        except KeyError:
            raise ValueError(("Must provide `data` or ",
                              "`x.attrs = {'center': float, 'scale': float}"))
    else:
        center = data.mean()
        scale = data.std()
    return center + scale * x


def design_matrix(x, poly_order=0, include_const=True):
    """Return array of polynomial weight vectors.

    Parameters
    ----------
    x : (M,) array_like
        Vector of polynomial inputs.
    poly_order : int
        Highest-ordered term exponent.
    include_const : bool
        If True, include a first column of all ones.

    Returns
    -------
    result : (M, poly_order+1) ndarray
        A Vandermonde matrix of increasing powers of `x`.
    """
    if isinstance(x, TensorSharedVariable):
        start = 0 if include_const else 1
        return pm.math.stack([x**i for i in range(start, poly_order+1)], axis=1)
    else:
        x = np.asarray(x)
        try:
            out = np.vander(x, poly_order+1, increasing=True)
        except ValueError:
            raise ValueError(f"poly_order value '{poly_order}' is invalid")
        if not include_const:
            out = out[:, 1:]
        return out


def pad_knots(knots, k=3):
    """Repeat first and last knots `k` times."""
    knots = np.asarray(knots)
    return np.concatenate([np.repeat(knots[0], k),
                           knots,
                           np.repeat(knots[-1], k)])


def bspline_basis(t, x=None, k=3, padded_knots=False):
    """Create the B-spline basis matrix of coefficients.

    Parameters
    ----------
    t : array_like, shape (n+k+1,)
        internal knots
    x : array_like, optional
        points at which to evaluate the B-spline bases. If `x` is not given,
        a `scipy.interpolate.BSpline` object will be returned.
    k : int, optional, default=3
        B-spline order
    padded_knots : bool, optional, default=False
        if True, treat the input `t` as padded, otherwise, pad `t` with `k`
        each of the leading and trailing "border knot" values.

    Returns
    -------
    if `x` is given:
    B : ndarray, shape (x.shape, n+k+1)
        B-spline basis functions evaluated at the given points `x`. The last
        dimension is the number of knots.
    else:
    b : :obj:scipy.interpolate.BSpline
        B-spline basis function object with identity matrix as weights.
    """
    if not padded_knots:
        t = pad_knots(t, k)
    m = len(t) - k - 1
    c = np.eye(m)  # only activate one basis at a time
    b = BSpline(t, c, k, extrapolate=False)
    if x is None:
        return b
    else:
        B = b(x)
        B[np.isnan(B)] = 0.0
        return B


def coef_table(models, mnames=None, params=None, std=True):
    """Create a summary table of coefficients in each model.

    Parameters
    ----------
    models : list of `Quap`
        The models over which to summarize.
    mnames : list of str, optional
        Names of the models.
    params : list of str, optional
        Names of specific parameters to return.
    std : bool, optional
        If True, also return a table of standard deviations.

    Returns
    -------
    ct, cs : pd.DataFrame
        DataFrames of the coefficients and their standard deviations.
    """
    coefs = [m.coef for m in models]
    stds = [m.std for m in models]

    def transform_ct(ct, mnames=None, params=None, value_name='coef'):
        """Make coefficient table tidy for plotting"""
        if mnames is not None:
            ct.columns = mnames
        ct.index.name = 'param'
        ct.columns.name = 'model'
        # Use params to filter by indexed variables 'a__0', 'a__1', etc.
        # should result from passing params=['a']
        if params is not None:
            try:
                subtables = [ct.loc[params]]  # track each filtered table
            except KeyError:
                subtables = []
            for p in params:
                subtables.append(ct.filter(regex=f"^{p}__[0-9]+", axis=0))
            ct = pd.concat(subtables).drop_duplicates()
        ct = (ct.T  # organize by parameter, then model
                .melt(ignore_index=False, value_name=value_name)
                .set_index('param', append=True)
              )
        return ct

    ct = transform_ct(pd.concat(coefs, axis=1), mnames, params)
    if not std:
        return ct

    cs = transform_ct(pd.concat(stds, axis=1), mnames, params,
                      value_name='std')
    return pd.concat([ct, cs], axis=1)


# TODO by_model=True param? Swap index levels.
def plot_coef_table(ct, q=0.89, fignum=None):
    """Plot the table of coefficients from `sts.coef_table`.

    Parameters
    ----------
    ct : :obj:`CoefTable`
        Coefficient table output from `coef_table`.
    q : float in [0, 1], optional
        The probability interval to plot.
    ax : Axes, optional
        The Axes on which to plot.

    Returns
    -------
    fig, ax : Figure and Axes where the plot was made.
    """
    fig = plt.figure(fignum, clear=True, constrained_layout=True)
    if not fig.axes:
        ax = fig.add_subplot()
    else:
        ax = fig.axes[-1]  # take most recent

    # Leverage Seaborn for basic setup
    sns.pointplot(data=ct.reset_index(), x='coef', y='param', hue='model',
                  join=False, dodge=0.3, ax=ax)

    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    colors = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            if not np.ma.is_masked(x) and not np.ma.is_masked(y):
                x_coords.append(x)
                y_coords.append(y)
                colors.append(point_pair.get_facecolor())

    # Manually add the errorbars since we have std values already
    z = stats.norm.ppf(1 - (1 - q)/2)
    errs = 2 * ct['std'] * z  # ± err -> 2 * ...
    errs = errs.dropna()
    ax.errorbar(x_coords, y_coords, fmt=' ', xerr=errs, ecolor=colors)
    ax.axvline(0, ls='--', c='k', lw=1, alpha=0.5)
    return fig, ax


# -----------------------------------------------------------------------------
#         Utilities
# -----------------------------------------------------------------------------
logsumexp = _logsumexp


def frame_to_dataset(df):
    """Convert DataFrame to ArviZ DataSet by combinining columns with
    multi-dimensional parameters, e.g. β__0, β__1, ..., β__N into β (N,).
    """
    tf = df.copy()
    var_names = tf.columns.str.replace('__[0-9]+', '', regex=True).unique()
    the_dict = dict()
    for v in var_names:
        the_dict[v] = np.expand_dims(tf.filter(like=v).values, 0)
    return az.convert_to_dataset(the_dict)


def dataset_to_frame(ds):
    """Convert ArviZ DataSet to DataFrame by separating columns with
    multi-dimensional parameters, e.g. β (N,) into β__0, β__1, ..., β__N.
    """
    df = pd.DataFrame()
    for vname, da in ds.items():
        v = da.mean('chain').squeeze()  # remove chain dimension
        if v.ndim == 1:                 # only draw dimension
            df[vname] = v.values
        elif v.ndim > 1:
            df[_names_from_vec(vname, v.shape[1])] = v.values
        else:
            raise ValueError(f"{vname} has invalid dimension {v.ndim}.")
    return df


def _names_from_vec(vname, ncols):
    """Create a list of strings ['x__0', 'x__1', ..., 'x__``ncols``'],
    where 'x' is ``vname``."""
    # TODO case of 2D, etc. variables
    fmt = '02d' if ncols > 10 else 'd'
    return [f"{vname}__{i:{fmt}}" for i in range(ncols)]


# -----------------------------------------------------------------------------
#         IC Functions
# -----------------------------------------------------------------------------
def inference_data(model, post=None, var_names=None, eval_at=None, Ns=1000):
    """Prepare the inference data structure for the model.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object.
    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``quap`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.

    Returns
    -------
    result : xarray.Dataset
        Log likelihood for each of the ``var_names``. Each will be an array of
        size (Ns, N), for ``Ns`` samples of the posterior, and `N` data points.
    """
    if post is None:
        post = model.sample(Ns)  # DataFrame with ['α', 'βn__0', 'βn__1', ...]

    if eval_at is not None:
        for k, v in eval_at.items():
            model.model.set_data(k, v)

    return pm.compute_log_likelihood(
        idata=az.convert_to_inference_data(frame_to_dataset(post)),
        model=model.model,
        var_names=var_names,
        progressbar=False,
    )


def loglikelihood(model, post=None, var_names=None, eval_at=None, Ns=1000):
    """Compute the log-likelihood of the data, given the model.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object.
    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``quap`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.

    Returns
    -------
    result : xarray.Dataset
        Log likelihood for each of the ``var_names``. Each will be an array of
        size (Ns, N), for ``Ns`` samples of the posterior, and `N` data points.
    """
    return (
        inference_data(model, post, var_names, eval_at, Ns)
        .log_likelihood
        .mean('chain')
    )


def lppd(model=None, loglik=None, post=None, var_names=None, eval_at=None,
         Ns=1000):
    """Compute the log pointwise predictive density for a model.

    Parameters
    ----------
    quap : :obj:`Quap`
        The fitted model object. If ``loglik`` is not given, ``quap`` will be
        used to compute it.
    loglik : dict like {var_name: (Ns, N) ndarray}
        The log-likelihood of each desired output variable in the model, where
        `Ns` is the number of samples of the posterior, and `N` is the number
        of data points for that variable.

    .. note::
        Only one of ``quap`` or ``loglik`` may be given.

    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``quap`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.

    Returns
    -------
    result : dict like {var_name: lppd}
    """
    if model is None and loglik is None:
        raise ValueError('One of `quap` or `loglik` must be given!')

    if loglik is None:
        loglik = loglikelihood(
            model=model,
            post=post,
            var_names=var_names,
            eval_at=eval_at,
            Ns=Ns,
        )

    if var_names is None:
        var_names = loglik.keys()

    out = dict()
    for v in var_names:
        out[v] = logsumexp(loglik[v], axis=0) - np.log(Ns)
    return out


def WAIC(model=None, loglik=None, post=None, var_names=None, eval_at=None,
         Ns=1000, pointwise=False):
    """Compute the Widely Applicable Information Criteria for the model.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object. If ``loglik`` is not given, ``quap`` will be
        used to compute it.
    loglik : dict like {var_name: (Ns, N) ndarray}
        The log-likelihood of each desired output variable in the model, where
        `Ns` is the number of samples of the posterior, and `N` is the number
        of data points for that variable.

    .. note::
        Only one of ``quap`` or ``loglik`` may be given.

    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``quap`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.
    pointwise : bool
        If True, return a vector of length `N` for each output variable, where
        `N` is the number of data points for that variable.

    Returns
    -------
    result : dict like {var_name: WAIC}
        The WAIC
    """
    if model is None and loglik is None:
        raise ValueError('One of `quap` or `loglik` must be given!')

    if loglik is None:
        loglik = loglikelihood(
            model=model,
            post=post,
            var_names=var_names,
            eval_at=eval_at,
            Ns=Ns,
        )

    the_lppd = lppd(loglik=loglik, var_names=var_names)

    out = dict()
    for v in the_lppd:
        penalty = loglik[v].var(dim='draw').values
        waic_vec = -2 * (the_lppd[v] - penalty)
        n_cases = loglik[v].shape[1]
        std_err = (n_cases * np.var(waic_vec))**0.5
        lppd_w = the_lppd[v] if pointwise else the_lppd[v].sum()
        w = waic_vec if pointwise else waic_vec.sum()
        p = penalty if pointwise else penalty.sum()
        out[v] = dict(waic=w, lppd=lppd_w, penalty=p, std=std_err)
    return out


def LOOIS(model=None, idata=None, post=None, var_names=None, eval_at=None,
          Ns=1000, pointwise=False):
    """Compute the Pareto-smoothed Importance Sampling Leave-One-Out
    Cross-Validation score of the model.

    Parameters
    ----------
    quap : :obj:`Quap`
        The fitted model object.
    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``quap`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.
    pointwise : bool
        If True, return a vector of length `N` for each output variable, where
        `N` is the number of data points for that variable.

    Returns
    -------
    result : dict like {var_name: WAIC}
        The WAIC
    """
    if model is None and idata is None:
        raise ValueError('One of `quap` or `idata` must be given!')

    if idata is None:
        idata = inference_data(
            model=model,
            post=post,
            var_names=var_names,
            eval_at=eval_at,
            Ns=Ns
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        loo = az.loo(idata, pointwise=pointwise)

    return dict(
        PSIS=-2*loo.elpd_loo,  # == loo_list$estimates['looic', 'Estimate']
        lppd=loo.elpd_loo,
        penalty=loo.p_loo,     # == loo_list$p_loo
        std=2*loo.se,          # == loo_list$estimates['looic', 'SE']
    )


# "Globalize" nested function for parallelization.
# See <https://gist.github.com/EdwinChan/3c13d3a746bb3ec5082f>
@contextmanager
def globalized(func):
    namespace = sys.modules[func.__module__]
    name, qualname = func.__name__, func.__qualname__
    func.__name__ = func.__qualname__ = f"__{name}_{uuid.uuid4().hex}"
    setattr(namespace, func.__name__, func)
    try:
        yield
    finally:
        delattr(namespace, func.__name__)
        func.__name__, func.__qualname__ = name, qualname


def LOOCV(model, ind_var, obs_var, out_var, X_data, y_data,
          lno=1, pointwise=False, parallel=True, progressbar=True):
    """Compute the leave-one-out cross-validation score of the model.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object.
    ind_var : str
        The name of the independent data variable in the model.
    obs_var : str
        The name of the observed data variable in the model.
    out_var : str
        The name of the output variable in the model.
    X_data : array_like
        The input data.
    y_data : array_like
        The output data.
    lno : int
        Number of data points to leave out per iteration.
    pointwise : bool
        If True, return the score for each data point.
    parallel : bool
        If True, run the main loop in parallel.
    progressbar : bool
        If True, print a progress bar.

    Returns
    -------
    result : dict
        Dictionary of results with attributes:
        loocv : float or (N,) array
            The deviance-scaled score.
        lppd : float or (N,) array
            The least pointwise posterior density of the data.
        std : float
            An estimate of the standard error of the ``loocv`` score.
    """
    N = len(y_data)
    M = N // lno  # number of chunks of data

    X_list = np.split(X_data, M)
    y_list = np.split(y_data, M)

    def leave_out(data_list, i):
        """Return an array with one element of the list removed."""
        return np.concatenate([data_list[j] for j in range(M) if j != i])

    # Fit a model to each chunk of data
    def loocv_func(i):
        """Perform cross-validation on data chunk `i`."""
        # Train the model on the data without chunk i
        model.model.set_data(ind_var, leave_out(X_list, i))
        model.model.set_data(obs_var, leave_out(y_list, i))
        test_quap = quap(model=model.model)

        # Compute the LPPD on the left-out chunk of data
        return (
            lppd(
                model=test_quap,
                eval_at={ind_var: X_list[i], obs_var: y_list[i]}
            )
            [out_var]
        )

    # NOTE **WARNING** SLOW CODE
    if parallel:
        with globalized(loocv_func):
            if progressbar:
                lppd_list = process_map(
                    loocv_func,
                    range(M),
                    desc='LOOCV',
                    leave=False,
                    max_workers=4,
                    position=1,
                )
            else:
                with Pool(4) as pool:
                    pool.map(loocv_func, range(M))
    else:
        iters = range(M)
        if progressbar:
            from tqdm import tqdm
            iters = tqdm(iters, desc='LOOCV')
        lppd_list = [loocv_func(i) for i in iters]

    lppd_cv = np.array(lppd_list).squeeze()
    c = lppd_cv if pointwise else lppd_cv.sum()

    # mean of chunks, var of data
    var = lppd_cv.var() if lno == 1 else lppd_cv.mean(axis=0).var()
    std_err = (N * var)**0.5

    return dict(
        loocv=-2*c,
        lppd=c,
        std=std_err,
    )


# -----------------------------------------------------------------------------
#         Simulations
# -----------------------------------------------------------------------------
# (R code 7.17 - 7.19)
def sim_train_test(N=20, k=3, rho=np.r_[0.15, -0.4], b_sigma=100):
    r"""Simulate fitting a model of `k` parameters to `N` data points.

    The default data-generating (i.e. "true") model used is:

    .. math::
            y_i ~ \mathcal{N}(μ_i, 1)
            μ_i = α + β_1 x_{1,i} + β_2 x_{2,i}
            α   = 0
            β_1 =  0.15
            β_2 = -0.4

    If `k` is greater than 3, additional :math:`x_{j,i}` and corresponding
    :math:`\beta_j` values will be included in the model, but should have no
    effect on predictive power since the underlying data does not depend on any
    additional parameters.

    Parameters
    ----------
    N : int
        The number of simulated data points.
    k : int
        The number of parameters in the linear model, including the intercept.
    rho : 1-D array_like
        A vector of "true" parameter values, excluding the intercept term.
    b_sigma : float
        The model standard deviation of the slope terms.

    Returns
    -------
    result : dict
        'dev' : dict with keys {'train', 'test'}
            The deviance of the model for the train and test data.
        'model' : :obj:Quap
            The model itself.
    """
    Y_SIGMA = 1
    # NOTE this line is *required* for expected parallel behavior
    np.random.seed()

    # Define the dimensions of the "true" distribution to match the model
    n_dim = 1 + len(rho)
    if n_dim < k:
        n_dim = k

    # Generate train/test data
    # NOTE this method of creating the data obfuscates the underlying linear
    # model, but it is a clean way of accommodating varying parameter lengths.
    Rho = np.eye(n_dim)
    Rho[0, 1:len(rho)+1] = rho
    Rho[1:len(rho)+1, 0] = rho

    # >>> Rho
    # === array([[ 1.  ,  0.15, -0.4 ],
    #            [ 0.15,  1.  ,  0.  ],
    #            [-0.4 ,  0.  ,  1.  ]])
    #

    true_dist = stats.multivariate_normal(mean=np.zeros(n_dim), cov=Rho)
    X_train = true_dist.rvs(N)  # (N, k)
    X_test = true_dist.rvs(N)

    # Separate the inputs and outputs for readability
    y_train, X_train = X_train[:, 0], X_train[:, 1:]
    y_test, X_test = X_test[:, 0], X_test[:, 1:]

    # Define the training matrix
    mm_train = np.ones((N, 1))  # intercept term
    if k > 1:
        mm_train = np.c_[mm_train, X_train[:, :k-1]]

    # Build and fit the model to the training data
    with pm.Model():
        X = pm.MutableData('X', mm_train)
        obs = pm.MutableData('obs', y_train)
        α = pm.Normal('α', 0, b_sigma, shape=(1,))
        if k == 1:
            μ = pm.Deterministic('μ', α)
        else:
            βn = pm.Normal('βn', 0, b_sigma, shape=(k-1,))
            β = pm.math.concatenate([α, βn])
            μ = pm.Deterministic('μ', pm.math.dot(X, β))
        y = pm.Normal('y', μ, Y_SIGMA, observed=obs, shape=obs.shape)
        q = quap()

    # -------------------------------------------------------------------------
    #         Compute the Information Criteria
    # -------------------------------------------------------------------------
    # NOTE for more efficient computation, we could get the inference data once
    # to use for LOOIS, and then extract the log-likelihood for lppd and WAIC.
    # LOOIS.

    # Compute the lppd
    lppd_train = lppd(q)['y']

    # Compute the lppd with the test data
    mm_test = np.ones((N, 1))
    if k > 1:
        mm_test = np.c_[mm_test, X_test[:, :k-1]]

    # Compute the posterior and log-likelihood
    idata = inference_data(q, eval_at={'X': mm_test, 'obs': y_test})
    loglik = idata.log_likelihood.mean('chain')

    lppd_test = lppd(loglik=loglik)['y']

    # Compute the deviance
    res = pd.Series({('deviance', 'train'): -2 * np.sum(lppd_train),
                     ('deviance', 'test'): -2 * np.sum(lppd_test)})

    wx = WAIC(loglik=loglik)['y']
    lx = LOOIS(idata=idata)
    cx = LOOCV(
        model=q,
        ind_var='X',
        obs_var='obs',
        out_var='y',
        X_data=mm_train,
        y_data=y_train,
    )

    # Compile Results
    res[('WAIC', 'test')] = wx['waic']
    res[('WAIC', 'err')] = np.abs(wx['waic'] - res[('deviance', 'test')])

    res[('LOOIC', 'test')] = lx['PSIS']
    res[('LOOIC', 'err')] = np.abs(lx['PSIS'] - res[('deviance', 'test')])

    res[('LOOCV', 'test')] = cx['loocv']
    res[('LOOCV', 'err')] = np.abs(cx['loocv'] - res[('deviance', 'test')])

    return dict(res=res, model=q)


# =============================================================================
# =============================================================================
