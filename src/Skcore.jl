# Skcore contains the actual implementation of the Julia part of scikit-learn.
# It's a flat module, which simplifies the implementation.
# Sklearn defines the visible interface. This schema makes it easier
# change the public module structure.
module Skcore

using PyCall
using Parameters
using SymDict

include("sk_utils.jl")

@pyimport2 sklearn
@pyimport sklearn.base as sk_base

# These are the functions that should be implemented by estimators/transformers
api = [:fit!, :transform, :fit_transform!,
       :predict, :predict_proba, :predict_log_proba,
       :score_samples, :sample,
       :score, :decision_function, :clone, :set_params!, :get_params,
       :is_classifier, :is_pairwise, :get_feature_names,
       :inverse_transform]

# Not sure if we should export all the api
for f in api @eval(export $f) end

export @simple_model_constructor

macro import_api()
    # I wish `importall ..` worked
    esc(:(begin $([Expr(:import, :., :., f) for f in api]...) end))
end

abstract BaseEstimator

# Note that I don't know the rationale for the `safe` argument - cstjean Feb2016
clone(py_model::PyObject) = sklearn.clone(py_model, safe=true)
is_classifier(py_model::PyObject) = sk_base.is_classifier(py_model)

is_pairwise(py_model) = false # global default - override for specific models
is_pairwise(py_model::PyObject) =
    haskey(py_model, "_pairwise") ? py_model[:_pairwise] : false

################################################################################
# These functions are useful for defining estimators that do not themselves
# contain other estimators

function simple_get_params(estimator, param_names::Vector{Symbol})
    Dict([name => getfield(estimator, name)
          for name in param_names])
end

function simple_set_params!{T}(estimator::T, params; param_names=nothing)
    for (k, v) in params
        if param_names !== nothing && !(k in param_names)
            throw(ArgumentError("An estimator of type $T was passed the invalid hyper-parameter $k. Valid hyper-parameters: $param_names"))
        end
        setfield!(estimator, k, v)
    end
    estimator
end

simple_clone{T}(estimator::T) =
    T(; get_params(estimator)...)

"""
    @simple_estimator_constructor function SomeEstimator(; param_1=..., param_2=..., ...)
        ...
        SomeEstimator(param_1, param_2, ...)
    end

This macro helps to implement the scikit-learn protocol for simple estimators
(those that do not contain other estimators). It is used in front of the
definition of an outer constructor. In addition to defining the constructor
normally, it also defines the `set_params!`, `get_params` and `clone` methods
for this estimator.

The arguments accepted by the constructor should all be keyword arguments with
a default value. In other words, a user should be able to instantiate an
estimator without passing any arguments to it. The arguments should all
correspond to hyperparameters describing the model or the optimisation problem
the estimator tries to solve. These initial arguments (or parameters) are
always remembered by the estimator. Also note that they should not be
documented under the “Attributes” section, but rather under the “Parameters”
section for that estimator.

In addition, every keyword argument accepted by the constructor should
correspond to a field on the instance. The macro relies on this to find
the relevant fields to set on an estimator when doing model selection.

It is used in front of an outer constructor. The macro assumes that this
constructor's keyword arguments are all the estimator's hyperparameters (all
parameters set by the user, that do not depend on the training data). It will
define the `get_params`, `set_params!` and `clone` methods accordingly.

There should be no logic, not even input validation, and the parameters should
not be changed. The corresponding logic should be put where the parameters are
used, typically in fit! """
macro simple_model_constructor(function_definition)
    type_name, args, kwargs, body =
        parse_function_definition(function_definition)
    param_names = Symbol[parse_kwarg(kw)[1] for kw in kwargs]
    @assert isempty(args) "A @model_init function should accept only keyword arguments representing the model's hyper-parameters"
    :(begin
        Skcore.get_params(estimator::$(esc(type_name)); deep=true) =
            simple_get_params(estimator, $param_names)
        Skcore.set_params!(estimator::$(esc(type_name)); params...) =
            simple_set_params!(estimator, params; param_names=$param_names)
        Skcore.clone(estimator::$(esc(type_name))) =
            simple_clone(estimator)
        $(esc(:(function $type_name(; $(kwargs...))
            $(body...)
        end)))
    end)
end

################################################################################
# Defaults

fit_transform!(estimator::BaseEstimator, X, y=nothing; fit_kwargs...) =
    transform(fit!(estimator, X, y; fit_kwargs...), X)

function set_params!(estimator::BaseEstimator; params...) # from base.py
    # Simple optimisation to gain speed (inspect is slow)
    if isempty(params) return estimator end

    valid_params = get_params(estimator, deep=true)
    for (key, value) in params
        sp = split(string(key), "__"; limit=2)
        if length(sp) > 1
            name, sub_name = sp
            if !haskey(valid_params, name::AbstractString)
                throw(ArgumentError("Invalid parameter $name for estimator $estimator"))
            end
            sub_object = valid_params[name]
            set_params!(sub_object; kwargify(Dict(sub_name=>value))...)
        else
            TODO() # should be straight-forward
        end
    end
    estimator
end


################################################################################

# Julia => Python
api_map = Dict(:decision_function => :decision_function,
               :fit! => :fit,
               :fit_transform! => :fit_transform,
               :get_feature_names => :get_feature_names,
               :get_params => :get_params,
               :inverse_transform => :inverse_transform,
               :predict => :predict,
               :predict_proba => :predict_proba,
               :predict_log_proba => :predict_log_proba,
               :score_samples => :score_samples,
               :sample => :sample,
               :score => :score,
               :transform => :transform,
               :set_params! => :set_params)

for (jl_fun, py_fun) in api_map
    @eval $jl_fun(py_model::PyObject, args...; kwargs...) =
        py_model[$(Expr(:quote, py_fun))](args...; kwargs...)
end

################################################################################

imported_python_modules =
    Dict(:LinearModels => :linear_model,
         :Datasets => :datasets,
         :Preprocessing => :preprocessing)

for (jl_module, py_module) in imported_python_modules
    @eval @pyimport sklearn.$py_module as $jl_module
end



include("pipeline.jl")
include("scorer.jl")
include("cross_validation.jl")
include("grid_search.jl")


end
