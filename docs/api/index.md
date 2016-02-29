# API-INDEX


## MODULE: Sklearn

---

## Macros [Internal]

[@reexportsk(identifiers...)](Sklearn.md#macro___reexportsk.1)      reexportsk(identifiers...)

## MODULE: Skcore

---

## Methods [Exported]

[check_consistent_length(arrays...)](Skcore.md#method__check_consistent_length.1)  Check that all arrays have consistent first dimensions.

[decision_function(pip::Skcore.Pipeline,  X)](Skcore.md#method__decision_function.1)  Applies transforms to the data, and the decision_function method of

[decision_function(pip::Skcore.Pipeline,  X,  y)](Skcore.md#method__decision_function.2)  Applies transforms to the data, and the decision_function method of

[fit!(self::Skcore.FeatureUnion,  X)](Skcore.md#method__fit.1)  Fit all transformers using X.

[fit!(self::Skcore.FeatureUnion,  X,  y)](Skcore.md#method__fit.2)  Fit all transformers using X.

[get_feature_names(self::Skcore.FeatureUnion)](Skcore.md#method__get_feature_names.1)  Get feature names from all transformers.

[inverse_transform(self::Skcore.Pipeline,  X)](Skcore.md#method__inverse_transform.1)  Applies inverse transform to the data.

[score(pip::Skcore.Pipeline,  X)](Skcore.md#method__score.1)  Applies transforms to the data, and the score method of the

[score(pip::Skcore.Pipeline,  X,  y)](Skcore.md#method__score.2)  Applies transforms to the data, and the score method of the

[transform(self::Skcore.FeatureUnion,  X)](Skcore.md#method__transform.1)  Transform X separately by each transformer, concatenate results.

---

## Types [Exported]

[Skcore.Pipeline](Skcore.md#type__pipeline.1)  Pipeline of transforms with a final estimator.

---

## Macros [Exported]

[@pyimport2(expr)](Skcore.md#macro___pyimport2.1)      @pyimport2 sklearn: (decomposition, clone)

---

## Methods [Internal]

[_fit!(self::Skcore.BaseSearchCV,  X::AbstractArray{T, N},  y,  parameter_iterable)](Skcore.md#method___fit.1)  Actual fitting,  performing the search over parameters.

[_fit_and_score(estimator,  X,  y,  scorer,  train::Array{Int64, 1},  test::Array{Int64, 1},  verbose,  parameters,  fit_params::Union{Dict{Symbol, Any}, Void})](Skcore.md#method___fit_and_score.1)  Fit estimator and compute scores for a given dataset split.

[_name_estimators(estimators)](Skcore.md#method___name_estimators.1)  Generate names for estimators.

[_safe_split(estimator,  X,  y,  indices)](Skcore.md#method___safe_split.1)  Create subset of dataset and properly handle kernels.

[_safe_split(estimator,  X,  y,  indices,  train_indices)](Skcore.md#method___safe_split.2)  Create subset of dataset and properly handle kernels.

[_score(estimator,  X_test,  y_test,  scorer)](Skcore.md#method___score.1)  Compute the score of an estimator on a given test set.

[check_scoring{T}(estimator::T)](Skcore.md#method__check_scoring.1)  Determine scorer from user options.

[check_scoring{T}(estimator::T,  scoring)](Skcore.md#method__check_scoring.2)  Determine scorer from user options.

[cross_val_score(estimator,  X)](Skcore.md#method__cross_val_score.1)  Evaluate a score by cross-validation

[cross_val_score(estimator,  X,  y)](Skcore.md#method__cross_val_score.2)  Evaluate a score by cross-validation

[kwargify(assoc::Associative{K, V})](Skcore.md#method__kwargify.1)   Turns `Dict("x"=>10, "y"=>40)` into `Dict(:x=>10, :y=>40)` 

[make_pipeline(steps...)](Skcore.md#method__make_pipeline.1)  Construct a Pipeline from the given estimators.

[make_scorer(score_func)](Skcore.md#method__make_scorer.1)  Make a scorer from a performance metric or loss function.

[pretransform(pip::Skcore.Pipeline,  X)](Skcore.md#method__pretransform.1)      pretransform(pip::Pipeline, X)

---

## Types [Internal]

[Skcore.FeatureUnion](Skcore.md#type__featureunion.1)  Concatenates results of multiple transformer objects.

[Skcore.GridSearchCV](Skcore.md#type__gridsearchcv.1)  Exhaustive search over specified parameter values for an estimator.

[Skcore.PredictScorer](Skcore.md#type__predictscorer.1)  Evaluate predicted target values for X relative to y_true.

