# Adapted from scikit-learn
# Copyright (c) 2007–2016 The scikit-learn developers.

# Original Python authors: Edouard Duchesnay
#                          Gael Varoquaux
#                          Virgile Fritsch
#                          Alexandre Gramfort
#                          Lars Buitinck
# Julia adaptation: Cedric St-Jean
# License: BSD


## """ The :mod:`sklearn.pipeline` module implements utilities to build a composite
## estimator, as a chain of transforms and estimators. """
## module Pipelines

## using ..nunique, ..BaseEstimator, ..@import_api, ..kwargify
## @import_api()

export Pipeline, named_steps, make_pipeline

"""Pipeline of transforms with a final estimator.

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
    an estimator. """
type Pipeline <: CompositeEstimator
    steps::Vector{Tuple{AbstractString, Any}} # of tuples(string, model)
    models::Vector{Any}
    function Pipeline(steps)
        @assert(nunique(map(first, steps)) == length(steps),
                "Pipeline's models' names must all be unique")
        new(steps, map(x->x[2], steps))
    end
end

get_models(pip::Pipeline) = pip.models
get_transforms(pip::Pipeline) = view(get_models(pip), 1:length(pip)-1)
get_final_estimator(pip::Pipeline) = get_models(pip)[end]
named_steps(pip::Pipeline) = Dict(pip.steps)
Base.length(pip::Pipeline) = length(pip.steps)

is_classifier(pip::Pipeline) = is_classifier(get_final_estimator(pip))
clone(pip::Pipeline) =
    Pipeline([(name, clone(model)) for (name, model) in pip.steps])

function fit!(pip::Pipeline, X, y=nothing)
    for tr in get_transforms(pip)
        # sklearn passes the y target to the transforms, and I'm not sure
        # why, but some of the rationale is here:
        # https://github.com/scikit-learn/scikit-learn/pull/3113
        # We could support transforms on y fairly easily if we
        # used an interface with the default:
        #    transform_xy(model, X, y) = (transform(model, X), y)
        X = fit_transform!(tr, X, y)
    end
    est = get_final_estimator(pip)
    fit!(est, X, y)
    pip
end

"""    pretransform(pip::Pipeline, X)

Applies all transformations (but not the final estimator) to `X` """
function pretransform(pip::Pipeline, X)
    Xt = X
    for transf in get_transforms(pip)
        Xt = transform(transf, Xt)
    end
    return Xt
end

predict(pip::Pipeline, X) =
    return predict(get_final_estimator(pip), pretransform(pip, X))
predict_proba(pip::Pipeline, X) =
    return predict_proba(get_final_estimator(pip), pretransform(pip, X))
predict_log_proba(pip::Pipeline, X) =
    return predict_log_proba(get_final_estimator(pip), pretransform(pip, X))
transform(pip::Pipeline, X) =
    return transform(get_final_estimator(pip), pretransform(pip, X))


"""Applies inverse transform to the data.
Starts with the last step of the pipeline and applies ``inverse_transform`` in
inverse order of the pipeline steps.
Valid only if all steps of the pipeline implement inverse_transform.

Parameters
----------
X : iterable
    Data to inverse transform. Must fulfill output requirements of the
    last step of the pipeline.
"""
function inverse_transform(self::Pipeline, X)
    if ndims(X) == 1
        X = X'
    end
    Xt = X
    for (name, step) in self.steps[end:-1:1]
        Xt = inverse_transform(step, Xt)
    end
    return Xt
end


function get_params(pip::Pipeline; deep=true)
    if !deep
        return Dict("steps"=>pip.steps)
    else
        out = copy(named_steps(pip))
        # Julia note: could probably just be pip.steps instead of named_steps
        for (name, step) in named_steps(pip)
            for (key, value) in get_params(step, deep=true)
                out["$(name)__$key"] = value
            end
        end
        return out
    end
end

"""Applies transforms to the data, and the score method of the
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
"""
score(pip::Pipeline, X, y=nothing) =
    return score(get_final_estimator(pip), pretransform(pip, X), y)


"""Applies transforms to the data, and the decision_function method of
the final estimator. Valid only if the final estimator implements
decision_function.

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step of
    the pipeline.
"""
decision_function(pip::Pipeline, X, y=nothing) =
    return decision_function(get_final_estimator(pip), pretransform(pip, X))


################################################################################
# FeatureUnion

"""Concatenates results of multiple transformer objects.

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

"""
type FeatureUnion <: CompositeEstimator
    transformer_list::Vector{Tuple{Any, Any}}
    n_jobs::Int
    transformer_weights
    FeatureUnion(transformer_list; n_jobs=1, transformer_weights=nothing) =
        new(transformer_list, n_jobs, transformer_weights)
end


clone(fu::FeatureUnion) = 
    FeatureUnion([(name, clone(model))
                  for (name, model) in fu.transformer_list];
                 n_jobs=fu.n_jobs, transformer_weights=fu.transformer_weights)


"""Get feature names from all transformers.

Returns
-------
feature_names : list of strings
    Names of the features produced by transform.
"""
function get_feature_names(self::FeatureUnion)
    feature_names = Any[]
    for (name, trans) in self.transformer_list
        push!(feature_names, [name * "__" * f for f in
                              get_feature_names(trans)])
    end
    return feature_names
end


"""Fit all transformers using X.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    Input data, used to fit transformers.
"""
function fit!(self::FeatureUnion, X, y=nothing)
    @assert self.n_jobs == 1 "n_jobs > 1 not supported yet. TODO"
    transformers = [fit!(trans, X, y)
                    for (name, trans) in self.transformer_list]
    _update_transformer_list(self, transformers)
    return self
end


# TODO: translate this
    ## def fit_transform(self, X, y=None, **fit_params):
    ##     """Fit all transformers using X, transform the data and concatenate
    ##     results.

    ##     Parameters
    ##     ----------
    ##     X : array-like or sparse matrix, shape (n_samples, n_features)
    ##         Input data to be transformed.

    ##     Returns
    ##     -------
    ##     X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
    ##         hstack of results of transformers. sum_n_components is the
    ##         sum of n_components (output dimension) over transformers.
    ##     """
    ##     result = Parallel(n_jobs=self.n_jobs)(
    ##         delayed(_fit_transform_one)(trans, name, X, y,
    ##                                     self.transformer_weights, **fit_params)
    ##         for name, trans in self.transformer_list)

    ##     Xs, transformers = zip(*result)
    ##     self._update_transformer_list(transformers)
    ##     if any(sparse.issparse(f) for f in Xs):
    ##         Xs = sparse.hstack(Xs).tocsr()
    ##     else:
    ##         Xs = np.hstack(Xs)
    ##     return Xs


function _transform_one(transformer, name, X, transformer_weights)
    if transformer_weights !== nothing
        # Not taking any chance until I know what objects come this way -cstjean
        @assert keytype(transformer_weights) == typeof(name)
    end
    if transformer_weights !== nothing && haskey(transformer_weights, name)
        # if we have a weight for this transformer, muliply output
        return transform(transformer, X) * transformer_weights[name]
    end
    return transform(transformer, X)
end


"""Transform X separately by each transformer, concatenate results.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    Input data to be transformed.

Returns
-------
X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
    hstack of results of transformers. sum_n_components is the
    sum of n_components (output dimension) over transformers.
"""
function transform(self::FeatureUnion, X)
    @assert self.n_jobs==1 "TODO: n_jobs>1"
    Xs = [_transform_one(trans, name, X, self.transformer_weights)
          for (name, trans) in self.transformer_list]
    if any(issparse, Xs)
        # I would just like a test case for it.
        error("TODO: sparse matrices not supported in FeatureUnions - not hard")
        #Xs = sparse.hstack(Xs).tocsr()
    else
        Xs = hcat(Xs...)
    end
    return Xs
end

function get_params(self::FeatureUnion; deep=true)
    if !deep
        return Dict("n_jobs" => self.n_jobs,
                    "transformer_weights" => self.transformer_weights,
                    "transformer_list" => self.transformer_list)
    else
        out = Dict(self.transformer_list)
        for (name, trans) in self.transformer_list
            for (key, value) in get_params(trans, deep=true)
                out["$(name)__$key"] = value
            end
        end
        return out
    end
end

function _update_transformer_list(self::FeatureUnion, transformers)
    self.transformer_list[:] =
        [(name, new)
         for ((name, old), new) in zip(self.transformer_list, transformers)]
end


get_classes(pip::Pipeline) = get_classes(get_final_estimator(pip))


################################################################################
# Convenience constructors

estimator_typename(est) = string(typeof(est))
estimator_typename(est::PyObject) = pytypeof(est)[:__name__]

"""Generate names for estimators."""
function _name_estimators(estimators)
    names = [lowercase(estimator_typename(estimator))
             for estimator in estimators]
    namecount = Dict()
    for (est, name) in zip(estimators, names)
        namecount[name] = get(namecount, name, 0) + 1
    end

    for (k, v) in namecount
        if v == 1
            delete!(namecount, k)
        end
    end

    for i in length(estimators):-1:1
        name = names[i]
        if haskey(namecount, name)
            names[i] *= "-$(namecount[name])"
            namecount[name] -= 1
        end
    end
    return collect(zip(names, estimators))
end


"""Construct a Pipeline from the given estimators.

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
"""
function make_pipeline(steps...)
    return Pipeline(_name_estimators(steps))
end

# This is an unexported hack for my own personal use - cstjean
dummy_input(pip::Pipeline) =
    dummy_input(first(get_transforms(pip)))

function fit_to_constant!(pip::Pipeline, constant)
    X = dummy_input(pip)
    y = fill(constant, size(X, 1))
    return fit!(pip, X, y)
end
## end
