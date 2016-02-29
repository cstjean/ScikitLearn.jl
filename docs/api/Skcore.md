# Skcore

## Exported

---

<a id="method__check_consistent_length.1" class="lexicon_definition"></a>
#### check_consistent_length(arrays...) [¶](#method__check_consistent_length.1)
Check that all arrays have consistent first dimensions.

Checks whether all objects in arrays have the same shape or length.

Parameters
----------
*arrays : list or tuple of input objects.
    Objects that will be checked for consistent length.


*source:*
[/Users/cedric/Programa/Sklearn/src/sk_utils.jl:62](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/sk_utils.jl#L62)

---

<a id="method__decision_function.1" class="lexicon_definition"></a>
#### decision_function(pip::Skcore.Pipeline,  X) [¶](#method__decision_function.1)
Applies transforms to the data, and the decision_function method of
the final estimator. Valid only if the final estimator implements
decision_function.

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step of
    the pipeline.


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:156](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L156)

---

<a id="method__decision_function.2" class="lexicon_definition"></a>
#### decision_function(pip::Skcore.Pipeline,  X,  y) [¶](#method__decision_function.2)
Applies transforms to the data, and the decision_function method of
the final estimator. Valid only if the final estimator implements
decision_function.

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step of
    the pipeline.


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:156](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L156)

---

<a id="method__fit.1" class="lexicon_definition"></a>
#### fit!(self::Skcore.FeatureUnion,  X) [¶](#method__fit.1)
Fit all transformers using X.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    Input data, used to fit transformers.


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:222](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L222)

---

<a id="method__fit.2" class="lexicon_definition"></a>
#### fit!(self::Skcore.FeatureUnion,  X,  y) [¶](#method__fit.2)
Fit all transformers using X.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    Input data, used to fit transformers.


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:222](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L222)

---

<a id="method__get_feature_names.1" class="lexicon_definition"></a>
#### get_feature_names(self::Skcore.FeatureUnion) [¶](#method__get_feature_names.1)
Get feature names from all transformers.

Returns
-------
feature_names : list of strings
    Names of the features produced by transform.


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:205](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L205)

---

<a id="method__inverse_transform.1" class="lexicon_definition"></a>
#### inverse_transform(self::Skcore.Pipeline,  X) [¶](#method__inverse_transform.1)
Applies inverse transform to the data.
Starts with the last step of the pipeline and applies ``inverse_transform`` in
inverse order of the pipeline steps.
Valid only if all steps of the pipeline implement inverse_transform.

Parameters
----------
X : iterable
    Data to inverse transform. Must fulfill output requirements of the
    last step of the pipeline.


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:101](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L101)

---

<a id="method__score.1" class="lexicon_definition"></a>
#### score(pip::Skcore.Pipeline,  X) [¶](#method__score.1)
Applies transforms to the data, and the score method of the
final estimator. Valid only if the final estimator implements
score.

Parameters
----------
X : iterable
    Data to score. Must fulfill input requirements of first step of the
    pipeline.

y : iterable, default=None
    Targets used for scoring. Must fulfill label requirements for all steps of
    the pipeline.


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:142](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L142)

---

<a id="method__score.2" class="lexicon_definition"></a>
#### score(pip::Skcore.Pipeline,  X,  y) [¶](#method__score.2)
Applies transforms to the data, and the score method of the
final estimator. Valid only if the final estimator implements
score.

Parameters
----------
X : iterable
    Data to score. Must fulfill input requirements of first step of the
    pipeline.

y : iterable, default=None
    Targets used for scoring. Must fulfill label requirements for all steps of
    the pipeline.


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:142](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L142)

---

<a id="method__transform.1" class="lexicon_definition"></a>
#### transform(self::Skcore.FeatureUnion,  X) [¶](#method__transform.1)
Transform X separately by each transformer, concatenate results.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    Input data to be transformed.

Returns
-------
X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
    hstack of results of transformers. sum_n_components is the
    sum of n_components (output dimension) over transformers.


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:287](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L287)

---

<a id="type__pipeline.1" class="lexicon_definition"></a>
#### Skcore.Pipeline [¶](#type__pipeline.1)
Pipeline of transforms with a final estimator.

Sequentially apply a list of transforms and a final estimator.
Intermediate steps of the pipeline must be 'transforms', that is, they
must implement fit! and transform methods.
The final estimator only needs to implement fit!.
The purpose of the pipeline is to assemble several steps that can be
cross-validated together while setting different parameters.

Read more in the :ref:`User Guide <pipeline>`.

Parameters
----------
steps : vector
    List of (name, transform) tuples (implementing fit/transform) that are
    chained, in the order in which they are chained, with the last object
    an estimator. 

*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:36](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L36)

---

<a id="macro___pyimport2.1" class="lexicon_definition"></a>
#### @pyimport2(expr) [¶](#macro___pyimport2.1)
    @pyimport2 sklearn: (decomposition, clone)

is the same as the Python code:

from sklearn import decomposition, clone



*source:*
[/Users/cedric/Programa/Sklearn/src/sk_utils.jl:15](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/sk_utils.jl#L15)

## Internal

---

<a id="method___fit.1" class="lexicon_definition"></a>
#### _fit!(self::Skcore.BaseSearchCV,  X::AbstractArray{T, N},  y,  parameter_iterable) [¶](#method___fit.1)
Actual fitting,  performing the search over parameters.

*source:*
[/Users/cedric/Programa/Sklearn/src/grid_search.jl:24](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/grid_search.jl#L24)

---

<a id="method___fit_and_score.1" class="lexicon_definition"></a>
#### _fit_and_score(estimator,  X,  y,  scorer,  train::Array{Int64, 1},  test::Array{Int64, 1},  verbose,  parameters,  fit_params::Union{Dict{Symbol, Any}, Void}) [¶](#method___fit_and_score.1)
Fit estimator and compute scores for a given dataset split.

Parameters
----------
estimator : estimator object implementing 'fit'
    The object to use to fit the data.

X : array-like of shape at least 2D
    The data to fit.

y : array-like, optional, default: nothing
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

parameters : dict or nothing
    Parameters to be set on the estimator.

fit_params : dict or nothing
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

parameters : dict or nothing, optional
    The parameters that have been evaluated.


*source:*
[/Users/cedric/Programa/Sklearn/src/cross_validation.jl:221](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/cross_validation.jl#L221)

---

<a id="method___name_estimators.1" class="lexicon_definition"></a>
#### _name_estimators(estimators) [¶](#method___name_estimators.1)
Generate names for estimators.

*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:334](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L334)

---

<a id="method___safe_split.1" class="lexicon_definition"></a>
#### _safe_split(estimator,  X,  y,  indices) [¶](#method___safe_split.1)
Create subset of dataset and properly handle kernels.

*source:*
[/Users/cedric/Programa/Sklearn/src/cross_validation.jl:288](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/cross_validation.jl#L288)

---

<a id="method___safe_split.2" class="lexicon_definition"></a>
#### _safe_split(estimator,  X,  y,  indices,  train_indices) [¶](#method___safe_split.2)
Create subset of dataset and properly handle kernels.

*source:*
[/Users/cedric/Programa/Sklearn/src/cross_validation.jl:288](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/cross_validation.jl#L288)

---

<a id="method___score.1" class="lexicon_definition"></a>
#### _score(estimator,  X_test,  y_test,  scorer) [¶](#method___score.1)
Compute the score of an estimator on a given test set.

*source:*
[/Users/cedric/Programa/Sklearn/src/cross_validation.jl:336](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/cross_validation.jl#L336)

---

<a id="method__check_scoring.1" class="lexicon_definition"></a>
#### check_scoring{T}(estimator::T) [¶](#method__check_scoring.1)
Determine scorer from user options.

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


*source:*
[/Users/cedric/Programa/Sklearn/src/cross_validation.jl:116](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/cross_validation.jl#L116)

---

<a id="method__check_scoring.2" class="lexicon_definition"></a>
#### check_scoring{T}(estimator::T,  scoring) [¶](#method__check_scoring.2)
Determine scorer from user options.

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


*source:*
[/Users/cedric/Programa/Sklearn/src/cross_validation.jl:116](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/cross_validation.jl#L116)

---

<a id="method__cross_val_score.1" class="lexicon_definition"></a>
#### cross_val_score(estimator,  X) [¶](#method__cross_val_score.1)
Evaluate a score by cross-validation

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


*source:*
[/Users/cedric/Programa/Sklearn/src/cross_validation.jl:69](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/cross_validation.jl#L69)

---

<a id="method__cross_val_score.2" class="lexicon_definition"></a>
#### cross_val_score(estimator,  X,  y) [¶](#method__cross_val_score.2)
Evaluate a score by cross-validation

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


*source:*
[/Users/cedric/Programa/Sklearn/src/cross_validation.jl:69](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/cross_validation.jl#L69)

---

<a id="method__kwargify.1" class="lexicon_definition"></a>
#### kwargify(assoc::Associative{K, V}) [¶](#method__kwargify.1)
 Turns `Dict("x"=>10, "y"=>40)` into `Dict(:x=>10, :y=>40)` 

*source:*
[/Users/cedric/Programa/Sklearn/src/sk_utils.jl:71](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/sk_utils.jl#L71)

---

<a id="method__make_pipeline.1" class="lexicon_definition"></a>
#### make_pipeline(steps...) [¶](#method__make_pipeline.1)
Construct a Pipeline from the given estimators.

This is a shorthand for the Pipeline constructor; it does not require, and
does not permit, naming the estimators. Instead, they will be given names
automatically based on their types.

Examples
--------
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.preprocessing import StandardScaler
>>> make_pipeline(StandardScaler(), GaussianNB())    # doctest: +NORMALIZE_WHITESPACE
Pipeline(steps=[('standardscaler',
                 StandardScaler(copy=True, with_mean=True, with_std=True)),
                ('gaussiannb', GaussianNB())])

Returns
-------
p : Pipeline


*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:378](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L378)

---

<a id="method__make_scorer.1" class="lexicon_definition"></a>
#### make_scorer(score_func) [¶](#method__make_scorer.1)
Make a scorer from a performance metric or loss function.

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
>>> from sklearn.metrics import fbeta_score, make_scorer
>>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
>>> ftwo_scorer
make_scorer(fbeta_score, beta=2)
>>> from sklearn.grid_search import GridSearchCV
>>> from sklearn.svm import LinearSVC
>>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
...                     scoring=ftwo_scorer)


*source:*
[/Users/cedric/Programa/Sklearn/src/scorer.jl:116](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/scorer.jl#L116)

---

<a id="method__pretransform.1" class="lexicon_definition"></a>
#### pretransform(pip::Skcore.Pipeline,  X) [¶](#method__pretransform.1)
    pretransform(pip::Pipeline, X)

Applies all transformations (but not the final estimator) to `X` 

*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:72](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L72)

---

<a id="type__featureunion.1" class="lexicon_definition"></a>
#### Skcore.FeatureUnion [¶](#type__featureunion.1)
Concatenates results of multiple transformer objects.

This estimator applies a list of transformer objects in parallel to the
input data, then concatenates the results. This is useful to combine
several feature extraction mechanisms into a single transformer.

Parameters
----------
transformer_list: list of (string, transformer) tuples
    List of transformer objects to be applied to the data. The first
    half of each tuple is the name of the transformer.

n_jobs: int, optional
    Number of jobs to run in parallel (default 1).

transformer_weights: dict, optional
    Multiplicative weights for features per transformer.
    Keys are transformer names, values the weights.



*source:*
[/Users/cedric/Programa/Sklearn/src/pipeline.jl:183](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/pipeline.jl#L183)

---

<a id="type__gridsearchcv.1" class="lexicon_definition"></a>
#### Skcore.GridSearchCV [¶](#type__gridsearchcv.1)
Exhaustive search over specified parameter values for an estimator.

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



*source:*
[/Users/cedric/Programa/Sklearn/src/grid_search.jl:259](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/grid_search.jl#L259)

---

<a id="type__predictscorer.1" class="lexicon_definition"></a>
#### Skcore.PredictScorer [¶](#type__predictscorer.1)
Evaluate predicted target values for X relative to y_true.

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


*source:*
[/Users/cedric/Programa/Sklearn/src/scorer.jl:28](https://github.com/cstjean/scikit-learn.jl/tree/81d6f1b0418104e79dbec21c96619d1998b13bba/src/scorer.jl#L28)

