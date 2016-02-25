# TODO: translate cross_val_predict

@pyimport2 sklearn.cross_validation: (_check_cv, check_cv, 
                                      _index_param_value)
## @pyimport2 sklearn.metrics: get_scorer


# Python indices are 0-based, so we need to transform the cross-validation
# iterators by adding 1 to each index.
# This code is rather dangerous, since if the caller had called `collect` on
# `cv`, the +1 would not be applied.
# I think the best option going forward would be to wrap the Python CV iterators
# to add the +1 near the source.
fix_cv_iter_indices(cv) = cv
fix_cv_iter_indices(cv::PyObject) =
    [(train .+ 1, test .+ 1) for (train, test) in cv]


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

    cv = fix_cv_iter_indices(_check_cv(cv, X, y,
                                       classifier=is_classifier(estimator)))

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


"""Fit estimator and compute scores for a given dataset split.

Parameters
----------
estimator : estimator object implementing 'fit'
    The object to use to fit the data.

X : array-like of shape at least 2D
    The data to fit.

y : array-like, optional, default: None
    The target variable to try to predict in the case of
    supervised learning.

scorer : callable
    A scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

train : array-like, shape (n_train_samples,)
    Indices of training samples.

test : array-like, shape (n_test_samples,)
    Indices of test samples.

verbose : integer
    The verbosity level.

error_score : 'raise' (default) or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised. If a numeric value is given,
    FitFailedWarning is raised. This parameter does not affect the refit
    step, which will always raise the error.

parameters : dict or None
    Parameters to be set on the estimator.

fit_params : dict or None
    Parameters that will be passed to ``estimator.fit``.

return_train_score : boolean, optional, default: False
    Compute and return score on training set.

return_parameters : boolean, optional, default: False
    Return parameters that has been used for the estimator.

Returns
-------
train_score : float, optional
    Score on training set, returned only if `return_train_score` is `True`.

test_score : float
    Score on test set.

n_test_samples : int
    Number of test samples.

scoring_time : float
    Time spent for fitting and scoring in seconds.

parameters : dict or None, optional
    The parameters that have been evaluated.
"""
function _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                        parameters, fit_params; return_train_score=false,
                        return_parameters=false, error_score="raise")
    # Julia TODO
    @assert error_score == "raise" "error_score = $error_score not supported"
    if verbose > 1
        if parameters === nothing
            msg = "no parameters to be set"
        else
            msg = ""
            # Julia TODO: translate this
            ## msg = '%s' % (', '.join('%s=%s' % (k, v)
            ##               for k, v in parameters.items()))
        end
        println("[CV] $msg")
    end

    # Adjust length of sample weights
    fit_params = fit_params!==nothing ? fit_params : Dict()
    fit_params = Dict([k => _index_param_value(X, v, train)
                       for (k, v) in fit_params])

    if parameters !== nothing
        set_params(estimator; parameters...)
    end

    start_time = time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    try
        if y_train === nothing
            fit!(estimator, X_train; fit_params...)
        else
            fit!(estimator, X_train, y_train; fit_params...)
        end
    end # Julia TODO: Python has some meaningful error handling here
    test_score = _score(estimator, X_test, y_test, scorer)
    if return_train_score
        train_score = _score(estimator, X_train, y_train, scorer)
    end

    scoring_time = time() - start_time

    if verbose > 2
        # Julia TODO: verbosity - see Python code
    end

    ret = return_train_score ? [train_score] : Any[]
    push!(ret, test_score, size(X_test, 1), scoring_time)
    if return_parameters
        push!(ret, parameters)
    end
    return ret
end


"""Create subset of dataset and properly handle kernels."""
function _safe_split(estimator, X, y, indices, train_indices=None)
    # Julia note: I don't get what they mean here. FIXME
    ## if hasattr(estimator, 'kernel') and callable(estimator.kernel)
    ##     # cannot compute the kernel values with custom function
    ##     raise ValueError("Cannot use a custom kernel function. "
    ##                      "Precompute the kernel matrix instead.")

    # Julia note: not sure how to translate this either. It might be how
    # sklearn detects kernels?
    ## if not hasattr(X, "shape"):
    ##     if getattr(estimator, "_pairwise", False):
    ##         raise ValueError("Precomputed kernels or affinity matrices have "
    ##                          "to be passed as arrays or sparse matrices.")
    ##     X_subset = [X[idx] for idx in indices]
    ## else:
    if is_pairwise(estimator)
        # X is a precomputed square kernel matrix
        if size(X, 1) != size(X, 2)
            throw(ArgumentError("X should be a square kernel matrix"))
        end
        # Julia TODO: I want a test case before trying to translate
        # this indexing code
        TODO()
        ## if train_indices === nothing
        ##     X_subset = X[np.ix_(indices, indices)]
    ## else:
        ##     X_subset = X[np.ix_(indices, train_indices)]
    else
        X_subset = X[indices, :]
    end

    if y !== nothing
        y_subset = y[indices]
    else
        y_subset = nothing
    end

    return X_subset, y_subset
end


"""Compute the score of an estimator on a given test set."""
function _score(estimator, X_test, y_test, scorer)
    @show scorer
    if y_test === nothing
        score = scorer(estimator, X_test)
    else
        score = scorer(estimator, X_test, y_test)
    end
    if !isa(score, Number)
        throw(ValueError("scoring must return a number, got a $(typeof(score)) instead."))
    end
    return score
end
