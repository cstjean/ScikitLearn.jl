@pyimport2 sklearn.metrics: (mean_squared_error)

abstract BaseScorer


"""Evaluate predicted target values for X relative to y_true.

Parameters
----------
estimator : object
Trained estimator to use for scoring. Must implement the `predict`
method. The output of that is used to compute the score.

X : array-like or sparse matrix
Test data that will be fed to estimator.predict.

y_true : array-like
Gold standard target values for X.

sample_weight : array-like, optional (default=nothing)
Sample weights.

Returns
-------
score : float
Score function applied to prediction of estimator on X.
"""
type PredictScorer <: BaseScorer
    # score_func should ideally be ::Function, but we're still relying on the
    # Python functions. Not a big deal, but TODO
    score_func
    sign::Float64
    kwargs
end

function Base.call(self::PredictScorer, estimator, X, y_true;
                   sample_weight=nothing)
    y_pred = predict(estimator, X)
    if sample_weight !== nothing
        return self.sign .* self.score_func(y_true, y_pred;
                                            sample_weight=sample_weight,
                                            self.kwargs...)
    else
        return self.sign .* self.score_func(y_true, y_pred;
                                            self.kwargs...)
    end
end


#############################################################################

get_scorer(scoring) = scoring
get_scorer(scoring::AbstractString) =
# I'd like to disable scoring::AbstractString - TODO
#    throw(ArgumentError("get_scorer does not accept string arguments - try using a Symbol instead"))
    get_scorer(Symbol(scoring))

function get_scorer(scoring::Symbol)
    if haskey(SCORERS, scoring)
        return SCORERS[scoring]
    else
        throw(ArgumentError("$scoring is not a valid scoring value. Valid options are $(sorted(keys(SCORERS)))"))
    end
end


"""Make a scorer from a performance metric or loss function.

This factory function wraps scoring functions for use in GridSearchCV
and cross_val_score. It takes a score function, such as ``accuracy_score``,
``mean_squared_error``, ``adjusted_rand_index`` or ``average_precision``
and returns a callable that scores an estimator's output.

Parameters
----------
score_func : callable,
    Score function (or loss function) with signature
    ``score_func(y, y_pred, **kwargs)``.

greater_is_better : boolean, default=True
    Whether score_func is a score function (default), meaning high is good,
    or a loss function, meaning low is good. In the latter case, the
    scorer object will sign-flip the outcome of the score_func.

needs_proba : boolean, default=False
    Whether score_func requires predict_proba to get probability estimates
    out of a classifier.

needs_threshold : boolean, default=False
    Whether score_func takes a continuous decision certainty.
    This only works for binary classification using estimators that
    have either a decision_function or predict_proba method.

    For example ``average_precision`` or the area under the roc curve
    can not be computed using discrete predictions alone.

**kwargs : additional arguments
    Additional parameters to be passed to score_func.

Returns
-------
scorer : callable
    Callable object that returns a scalar score; greater is better.

Examples
--------
    from sklearn.metrics import fbeta_score, make_scorer
    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import LinearSVC
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
                        scoring=ftwo_scorer)
"""
function make_scorer(score_func; greater_is_better=true,
                     needs_proba=false, needs_threshold=false, kwargs...)
    sign = greater_is_better ? 1 : -1
    if needs_proba && needs_threshold
        throw(ArgumentError("Set either needs_proba or needs_threshold to True, but not both."))
    end
    if needs_proba
        cls = ProbaScorer
    elseif needs_threshold
        cls = ThresholdScorer
    else
        cls = PredictScorer
    end
    return cls(score_func, sign, kwargs)
end


const mean_squared_error_scorer = make_scorer(mean_squared_error,
                                              greater_is_better=false)

const SCORERS = Dict(
                     ## r2=r2_scorer,
                     ## median_absolute_error=median_absolute_error_scorer,
                     ## mean_absolute_error=mean_absolute_error_scorer,
                     :mean_squared_error=>mean_squared_error_scorer,
                     ## accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
                     ## average_precision=average_precision_scorer,
                     ## log_loss=log_loss_scorer,
                     ## adjusted_rand_score=adjusted_rand_scorer)
                     )
