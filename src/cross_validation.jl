# TODO: cross_val_predict

@pyimport2 sklearn.cross_validation: (_check_cv, check_cv, _fit_and_score)
@pyimport2 sklearn.metrics: get_scorer



"""Evaluate a score by cross-validation

Parameters
----------
estimator : estimator object implementing 'fit'
    The object to use to fit the data.

X : array-like
    The data to fit. Can be, for example a list, or an array at least 2d.

y : array-like, optional, default: nothing
    The target variable to try to predict in the case of
    supervised learning.

scoring : string, callable or nothing, optional, default: nothing
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

cv : cross-validation generator or int, optional, default: nothing
    A cross-validation generator to use. If int, determines
    the number of folds in StratifiedKFold if y is binary
    or multiclass and estimator is a classifier, or the number
    of folds in KFold otherwise. If nothing, it is equivalent to cv=3.

n_jobs : integer, optional
    The number of CPUs to use to do the computation. -1 means
    'all CPUs'.

verbose : integer, optional
    The verbosity level.

fit_params : dict, optional
    Parameters to pass to the fit method of the estimator.

Returns
-------
scores : array of float, shape=(len(list(cv)),)
    Array of scores of the estimator for each run of the cross validation.
"""
function cross_val_score(estimator, X, y=nothing; scoring=nothing, cv=nothing,
                         n_jobs=1, verbose=0, fit_params=nothing)
    # See the original source for inspiration on how to do it.
    @assert n_jobs==1 "Parallel cross-validation not supported yet. TODO"

    check_consistent_length(X, y)

    cv = _check_cv(cv, X, y, classifier=is_classifier(estimator))

    scorer = check_scoring(estimator, scoring)
    # We clone the estimator to make sure that all the folds are independent
    scores = Float64[_fit_and_score(clone(estimator), X, y, scorer,
                                    train, test, verbose, nothing,
                                    fit_params)[1]
                     for (train, test) in cv]

    return scores
end



"""Determine scorer from user options.

A TypeError will be thrown if the estimator cannot be scored.

Parameters
----------
estimator : estimator object implementing 'fit'
    The object to use to fit the data.

scoring : string, callable or nothing, optional, default: nothing
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

allow_none : boolean, optional, default: False
    If no scoring is specified and the estimator has no score function, we
    can either return nothing or raise an exception.

Returns
-------
scoring : callable
    A scorer callable object / function with signature
    ``scorer(estimator, X, y)``.
"""
function check_scoring{T}(estimator::T, scoring=nothing; allow_none=false):
    @assert !allow_none "TODO: allow_none=true"
    # sklearn asserts that estimator has a `fit` method. We should do that too.
    if scoring !== nothing
        return get_scorer(scoring)
    elseif true  # should be: if `score(::T)` is defined. Use `method_exists`
        return score
    elseif allow_none
        nothing
    else
        throw(TypeError("If no scoring is specified, the estimator passed should have a 'score' method. The estimator $estimator does not."))
    end
end


## """Input checker utility for building a CV in a user friendly way.

## Parameters
## ----------
## cv : int, a cv generator instance, or None
##     The input specifying which cv generator to use. It can be an
##     integer, in which case it is the number of folds in a KFold,
##     None, in which case 3 fold is used, or another object, that
##     will then be used as a cv generator.

## X : array-like
##     The data the cross-val object will be applied on.

## y : array-like
##     The target variable for a supervised learning problem.

## classifier : boolean optional
##     Whether the task is a classification task, in which case
##     stratified KFold will be used.

## Returns
## -------
## checked_cv: a cross-validation generator instance.
##     The return value is guaranteed to be a cv generator instance, whatever
##     the input type.
## """
## function check_cv(cv, X=nothing, y=nothing; classifier=true)
##     return _check_cv(cv, X=X, y=y, classifier=classifier, warn_mask=true)
## end


## function _check_cv(cv, X=nothing, y=nothing, classifier=false, warn_mask=false)
##     # This exists for internal use while indices is being deprecated.
##     is_sparse = sp.issparse(X)
##     needs_indices = is_sparse or not hasattr(X, "shape")
##     if cv is None:
##         cv = 3
##     if isinstance(cv, numbers.Integral):
##         if warn_mask and not needs_indices:
##             warnings.warn('check_cv will return indices instead of boolean '
##                           'masks from 0.17', DeprecationWarning)
##         else:
##             needs_indices = None
##         if classifier:
##             if type_of_target(y) in ['binary', 'multiclass']:
##                 cv = StratifiedKFold(y, cv, indices=needs_indices)
##             else:
##                 cv = KFold(_num_samples(y), cv, indices=needs_indices)
##         else:
##             if not is_sparse:
##                 n_samples = len(X)
##             else:
##                 n_samples = X.shape[0]
##             cv = KFold(n_samples, cv, indices=needs_indices)
##     if needs_indices and not getattr(cv, "_indices", True):
##         raise ValueError("Sparse data and lists require indices-based cross"
##                          " validation generator, got: %r", cv)
##     return cv
