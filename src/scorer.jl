# Adapted from scikit-learn by Cedric St-Jean
# Copyright (c) 2007–2016 The scikit-learn developers.

using ScikitLearnBase: mean_squared_error
using StatsBase: Weights

@compat abstract type BaseScorer end


"""R^2 (coefficient of determination) regression score function.
Best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.
Read more in the :ref:`User Guide <r2_score>`.
Parameters
----------
y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.
y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.
sample_weight : array-like of shape = (n_samples), optional
    Sample weights.
multioutput : string in ['raw_values', 'uniform_average', \
'variance_weighted'] or None or array-like of shape (n_outputs)
    Defines aggregating of multiple output scores.
    Array-like value defines weights used to average scores.
    Default is "uniform_average".
    'raw_values' :
        Returns a full set of scores in case of multioutput input.
    'uniform_average' :
        Scores of all outputs are averaged with uniform weight.
    'variance_weighted' :
        Scores of all outputs are averaged, weighted by the variances
        of each individual output.
    .. versionchanged:: 0.19
        Default value of multioutput is 'uniform_average'.
Returns
-------
z : float or ndarray of floats
    The R^2 score or ndarray of scores if 'multioutput' is
    'raw_values'.
Notes
-----
This is not a symmetric function.
Unlike most other scores, R^2 score may be negative (it need not actually
be the square of a quantity R).
References
----------
.. [1] `Wikipedia entry on the Coefficient of determination
        <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
Examples
--------
>>> from sklearn.metrics import r2_score
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
0.948...
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> r2_score(y_true, y_pred, multioutput='variance_weighted')
... # doctest: +ELLIPSIS
0.938...
>>> y_true = [1,2,3]
>>> y_pred = [1,2,3]
>>> r2_score(y_true, y_pred)
1.0
>>> y_true = [1,2,3]
>>> y_pred = [2,2,2]
>>> r2_score(y_true, y_pred)
0.0
>>> y_true = [1,2,3]
>>> y_pred = [3,2,1]
>>> r2_score(y_true, y_pred)
-3.0
"""
function r2_score(y_true::AbstractVector, y_pred::AbstractVector;  sample_weight=nothing, multioutput="uniform_average")
	# y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)  # TODO
	if size(y_true)[1] != size(y_pred)[1]
		throw(DimensionMismatch("y_true and y_pred have different number of output $(size(y_true)[1]) >= $(size(y_pred)[1])"))
	end

	if sample_weight == nothing
		weight = 1.0
		sample_weight = Weights([weight for i in 1:size(y_true)[1]])
	else
		sample_weight = Weights(sample_weight)
	end

	numerator = sum(weight .* (y_true .- y_pred) .^ 2, 1)
	denominator = sum(weight .* (y_true .- mean(y_true, sample_weight, 1)) .^ 2, 1)

	nonzero_denominator = denominator != 0
	nonzero_numerator = numerator != 0
	valid_score = nonzero_denominator .& nonzero_numerator
	output_scores = Array{typeof(y_true[1])}(ndims(y_true))
	output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
	# arbitrary set to zero to avoid -inf scores, having a constant
	# y_true is not interesting for scoring a regression anyway
	for dd in 1:length(numerator)
		if nonzero_numerator[dd] & ~nonzero_denominator[dd]
			output_scores[dd] = 0.0
		end
	end
	if typeof(multioutput) <: String
		if multioutput == "raw_values"
			# return scores individually
			return output_scores
		elseif multioutput == "uniform_average"
			avg_weights = Weights([1.0 for i in 1:length(output_scores)])
		elseif multioutput == "variance_weighted"
			avg_weights = Weights(denominator)
			# avoid fail on constant y or one-element arrays
			if !any(nonzero_denominator)
				if !any(nonzero_numerator)
					return 1.0
				else
					return 0.0
				end
			end
		end
	else
		avg_weights = Weights(multioutput)
	end
	return mean(output_scores, avg_weights)
end

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

@compat function (self::PredictScorer)(estimator, X, y_true;
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
        throw(ArgumentError("$scoring is not a valid scoring value. Valid options are $(sort(collect(keys(SCORERS))))"))
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


const mean_squared_error_scorer = make_scorer(mean_squared_error, greater_is_better=false)
const r2_scorer = make_scorer(r2_score, greater_is_better=true)

const SCORERS = Dict(
                     :r2=>r2_scorer,
                     ## median_absolute_error=median_absolute_error_scorer,
                     ## mean_absolute_error=mean_absolute_error_scorer,
                     :mean_squared_error=>mean_squared_error_scorer,
                     ## accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
                     ## average_precision=average_precision_scorer,
                     ## log_loss=log_loss_scorer,
                     ## adjusted_rand_score=adjusted_rand_scorer)
                     )
