# Adapted from scikit-learn
# Copyright (c) 2007–2016 The scikit-learn developers.


## module GridSearch

## using PyCall
## using Parameters: @with_kw

## using ..nunique, ..BaseEstimator, ..@import_api, ..kwargify, ..@pyimport2
## using ..CrossValidation.check_scoring, ..CrossValidation.check_cv
## using ..CrossValidation._fit_and_score

abstract BaseSearchCV

@pyimport2 sklearn.grid_search: (ParameterGrid, _check_param_grid, Sized,
                                 ParameterSampler)

immutable CVScoreTuple
    parameters
    mean_validation_score
    cv_validation_scores
end


"""Actual fitting,  performing the search over parameters."""
function _fit!(self::BaseSearchCV, X::AbstractArray, y,
               parameter_iterable)
    estimator = self.estimator
    cv = self.cv
    self.scorer_ = check_scoring(self.estimator, self.scoring)

    n_samples = size(X, 1)

    if y !== nothing
        if size(y, 1) != n_samples
            throw(ArgumentError("Target variable (y) has a different number of samples ($(size(y, 1))) than data (X: $n_samples samples)"))
        end
    end
    cv = check_cv(cv, X, y,
                  classifier=is_classifier(estimator))

    if self.verbose > 0
        if pyisinstance(parameter_iterable, Sized)
            n_candidates = length(parameter_iterable)
            println("Fitting $(length(cv)) folds for each of $n_candidates candidates, totalling $(n_candidates * length(cv)) fits")
        end
    end


    base_estimator = clone(self.estimator)

    @assert self.n_jobs == 1 "TODO: support n_jobs > 1"
    out = vcat(Any[[_fit_and_score(clone(base_estimator), X, y, self.scorer_,
                                   train, test, self.verbose,
                                   kwargify(parameters),
                                   kwargify(self.fit_params),
                                   return_parameters=true,
                                   error_score=self.error_score)
                    for (train, test) in cv]
                   for parameters in parameter_iterable]...)

    # Out is a list of triplet: score, estimator, n_test_samples
    n_fits = length(out)
    n_folds = length(cv)

    scores = Tuple[]
    grid_scores = CVScoreTuple[]
    for grid_start in 1:n_folds:n_fits
        n_test_samples = 0
        score = 0
        all_scores = Float64[]
        _, _, _, current_parameters = out[grid_start]
        for (this_score, this_n_test_samples, _, parameters) in
            out[grid_start:grid_start + n_folds - 1]
            @assert parameters == current_parameters # same for the whole loop
            push!(all_scores, this_score)
            if self.iid
                this_score *= this_n_test_samples
                n_test_samples += this_n_test_samples
            end
            score += this_score
        end
        if self.iid
            score /= n_test_samples
        else
            score /= n_folds
        end
        push!(scores, (score, current_parameters))
        push!(grid_scores, CVScoreTuple(current_parameters, score, all_scores))
    end
    # Store the computed scores
    self.grid_scores_ = grid_scores

    # Find the best parameters by comparing on the mean validation score:
    # note that `sorted` is deterministic in the way it breaks ties
    best = sort(grid_scores, by=score_tup->score_tup.mean_validation_score,
                rev=true)[1]
    self.best_params_ = best.parameters
    self.best_score_ = best.mean_validation_score

    if self.refit
        # fit the best estimator using the entire dataset
        # clone first to work around broken estimators
        best_estimator = set_params!(clone(base_estimator); best.parameters...)
        if y !== nothing
            fit!(best_estimator, X, y; self.fit_params...)
        else
            fit!(best_estimator, X; self.fit_params...)
        end
        self.best_estimator_ = best_estimator
    end
    return self
end


"""Exhaustive search over specified parameter values for an estimator.

Important members are fit, predict.

GridSearchCV implements a "fit" method and a "predict" method like
any classifier except that the parameters of the classifier
used to predict is optimized by cross-validation.

Parameters
----------
estimator : object type that implements the "fit" and "predict" methods
    A object of that type is instantiated for each grid point.

param_grid : dict or list of dictionaries
    Dictionary with parameters names (string) as keys and lists of
    parameter settings to try as values, or a list of such
    dictionaries, in which case the grids spanned by each dictionary
    in the list are explored. This enables searching over any sequence
    of parameter settings.

scoring : string, callable or None, optional, default: None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

fit_params : dict, optional
    Parameters to pass to the fit method.

n_jobs : int, default 1
    Number of jobs to run in parallel.

pre_dispatch : int, or string, optional
    Controls the number of jobs that get dispatched during parallel
    execution. Reducing this number can be useful to avoid an
    explosion of memory consumption when more jobs get dispatched
    than CPUs can process. This parameter can be:

        - None, in which case all the jobs are immediately
          created and spawned. Use this for lightweight and
          fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs

        - An int, giving the exact number of total jobs that are
          spawned

        - A string, giving an expression as a function of n_jobs,
          as in '2*n_jobs'

iid : boolean, default=True
    If True, the data is assumed to be identically distributed across
    the folds, and the loss minimized is the total loss per sample,
    and not the mean loss across the folds.

cv : integer or cross-validation generator, default=3
    If an integer is passed, it is the number of folds.
    Specific cross-validation objects can be passed, see
    sklearn.cross_validation module for the list of possible objects

refit : boolean, default=True
    Refit the best estimator with the entire dataset.
    If "False", it is impossible to make predictions using
    this GridSearchCV instance after fitting.

verbose : integer
    Controls the verbosity: the higher, the more messages.

error_score : 'raise' (default) or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised. If a numeric value is given,
    FitFailedWarning is raised. This parameter does not affect the refit
    step, which will always raise the error.


Examples
--------
>>> from sklearn import svm, grid_search, datasets
>>> iris = datasets.load_iris()
>>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
>>> svr = svm.SVC()
>>> clf = grid_search.GridSearchCV(svr, parameters)
>>> clf.fit(iris.data, iris.target)
...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
GridSearchCV(cv=None, error_score=...,
       estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                     degree=..., gamma=..., kernel='rbf', max_iter=-1,
                     probability=False, random_state=None, shrinking=True,
                     tol=..., verbose=False),
       fit_params={}, iid=..., n_jobs=1,
       param_grid=..., pre_dispatch=..., refit=...,
       scoring=..., verbose=...)


Attributes
----------
grid_scores_ : list of named tuples
    Contains scores for all parameter combinations in param_grid.
    Each entry corresponds to one parameter setting.
    Each named tuple has the attributes:

        * ``parameters``, a dict of parameter settings
        * ``mean_validation_score``, the mean score over the
          cross-validation folds
        * ``cv_validation_scores``, the list of scores for each fold

best_estimator_ : estimator
    Estimator that was chosen by the search, i.e. estimator
    which gave highest score (or smallest loss if specified)
    on the left out data. Not available if refit=False.

best_score_ : float
    Score of best_estimator on the left out data.

best_params_ : dict
    Parameter setting that gave the best results on the hold out data.

scorer_ : function
    Scorer function used on the held out data to choose the best
    parameters for the model.

Notes
------
The parameters selected are those that maximize the score of the left out
data, unless an explicit score is passed in which case it is used instead.

If `n_jobs` was set to a value higher than one, the data is copied for each
point in the grid (and not `n_jobs` times). This is done for efficiency
reasons if individual jobs take very little time, but may raise errors if
the dataset is large and not enough memory is available.  A workaround in
this case is to set `pre_dispatch`. Then, the memory is copied only
`pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
n_jobs`.

See Also
---------
:class:`ParameterGrid`:
    generates all the combinations of a an hyperparameter grid.

:func:`sklearn.cross_validation.train_test_split`:
    utility function to split the data into a development set usable
    for fitting a GridSearchCV instance and an evaluation set for
    its final evaluation.

:func:`sklearn.metrics.make_scorer`:
    Make a scorer from a performance metric or loss function.

"""
@with_kw type GridSearchCV <: BaseSearchCV
    estimator
    param_grid
    scoring=nothing
    loss_func=nothing
    score_func=nothing
    fit_params=Dict()
    n_jobs=1
    iid=true
    refit=true
    cv=nothing
    verbose=0
    error_score="raise"
    # these are not parameters - they are set in _fit
    scorer_=nothing 
    best_params_=nothing
    best_score_=nothing
    grid_scores_=nothing
    best_estimator_=nothing
end
function GridSearchCV(estimator, param_grid; kwargs...)
    _check_param_grid(param_grid)
    GridSearchCV(estimator=estimator, param_grid=param_grid; kwargs...)
end

"""Run fit with all sets of parameters.

Parameters
----------

X : array-like, shape = [n_samples, n_features]
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like, shape = [n_samples] or [n_samples, n_output], optional
    Target relative to X for classification or regression;
    None for unsupervised learning.

"""
function fit!(self::GridSearchCV, X, y=nothing)
    return _fit!(self, X, y, ParameterGrid(self.param_grid))
end

is_classifier(gcv::GridSearchCV) = is_classifier(gcv.estimator)

################################################################################
# RandomizedSearchCV

"""Randomized search on hyper parameters.

RandomizedSearchCV implements a "fit" method and a "predict" method like
any classifier except that the parameters of the classifier
used to predict is optimized by cross-validation.

In contrast to GridSearchCV, not all parameter values are tried out, but
rather a fixed number of parameter settings is sampled from the specified
distributions. The number of parameter settings that are tried is
given by n_iter.

If all parameters are presented as a list,
sampling without replacement is performed. If at least one parameter
is given as a distribution, sampling with replacement is used.
It is highly recommended to use continuous distributions for continuous
parameters.

Parameters
----------
estimator : object type that implements the "fit" and "predict" methods
    A object of that type is instantiated for each parameter setting.

param_distributions : dict
    Dictionary with parameters names (string) as keys and distributions
    or lists of parameters to try. Distributions must provide a ``rvs``
    method for sampling (such as those from scipy.stats.distributions).
    If a list is given, it is sampled uniformly.

n_iter : int, default=10
    Number of parameter settings that are sampled. n_iter trades
    off runtime vs quality of the solution.

scoring : string, callable or None, optional, default: None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

fit_params : dict, optional
    Parameters to pass to the fit method.

n_jobs : int, default=1
    Number of jobs to run in parallel.

pre_dispatch : int, or string, optional
    Controls the number of jobs that get dispatched during parallel
    execution. Reducing this number can be useful to avoid an
    explosion of memory consumption when more jobs get dispatched
    than CPUs can process. This parameter can be:

        - None, in which case all the jobs are immediately
          created and spawned. Use this for lightweight and
          fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs

        - An int, giving the exact number of total jobs that are
          spawned

        - A string, giving an expression as a function of n_jobs,
          as in '2*n_jobs'

iid : boolean, default=True
    If True, the data is assumed to be identically distributed across
    the folds, and the loss minimized is the total loss per sample,
    and not the mean loss across the folds.

cv : integer or cross-validation generator, optional
    If an integer is passed, it is the number of folds (default 3).
    Specific cross-validation objects can be passed, see
    sklearn.cross_validation module for the list of possible objects

refit : boolean, default=True
    Refit the best estimator with the entire dataset.
    If "False", it is impossible to make predictions using
    this RandomizedSearchCV instance after fitting.

verbose : integer
    Controls the verbosity: the higher, the more messages.

error_score : 'raise' (default) or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised. If a numeric value is given,
    FitFailedWarning is raised. This parameter does not affect the refit
    step, which will always raise the error.


Attributes
----------
grid_scores_ : list of named tuples
    Contains scores for all parameter combinations in param_grid.
    Each entry corresponds to one parameter setting.
    Each named tuple has the attributes:

        * ``parameters``, a dict of parameter settings
        * ``mean_validation_score``, the mean score over the
          cross-validation folds
        * ``cv_validation_scores``, the list of scores for each fold

best_estimator_ : estimator
    Estimator that was chosen by the search, i.e. estimator
    which gave highest score (or smallest loss if specified)
    on the left out data. Not available if refit=False.

best_score_ : float
    Score of best_estimator on the left out data.

best_params_ : dict
    Parameter setting that gave the best results on the hold out data.

Notes
-----
The parameters selected are those that maximize the score of the held-out
data, according to the scoring parameter.

If `n_jobs` was set to a value higher than one, the data is copied for each
parameter setting(and not `n_jobs` times). This is done for efficiency
reasons if individual jobs take very little time, but may raise errors if
the dataset is large and not enough memory is available.  A workaround in
this case is to set `pre_dispatch`. Then, the memory is copied only
`pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
n_jobs`.

See Also
--------
:class:`GridSearchCV`:
    Does exhaustive search over a grid of parameters.

:class:`ParameterSampler`:
    A generator over parameter settins, constructed from
    param_distributions.

"""
@with_kw type RandomizedSearchCV <: BaseSearchCV
    estimator
    param_distributions
    n_iter=10
    scoring=nothing
    fit_params=Dict()
    n_jobs=1
    iid=true
    refit=true
    cv=nothing
    verbose=0
    random_state=nothing
    error_score="raise"

    scorer_=nothing
    grid_scores_=nothing
    best_estimator_=nothing
    best_score_=nothing
    best_params_=nothing
end

function RandomizedSearchCV(estimator, param_distributions; kwargs...)
    RandomizedSearchCV(estimator=estimator,
                       param_distributions=param_distributions; kwargs...)
end


"""Run fit on the estimator with randomly drawn parameters.

Parameters
----------
X : array-like, shape = [n_samples, n_features]
    Training vector, where n_samples in the number of samples and
    n_features is the number of features.

y : array-like, shape = [n_samples] or [n_samples, n_output], optional
    Target relative to X for classification or regression;
    None for unsupervised learning.

"""
function fit!(self::RandomizedSearchCV, X, y=nothing)
    sampled_params = ParameterSampler(self.param_distributions,
                                      self.n_iter,
                                      random_state=self.random_state)
    return _fit!(self, X, y, sampled_params)
end

## end
