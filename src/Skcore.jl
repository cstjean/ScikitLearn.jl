# Skcore contains the actual implementation of the Julia part of scikit-learn.
# It's a flat module, which simplifies the implementation.
# Sklearn defines the visible interface. This schema makes it easier
# change the public module structure.
module Skcore

using ScikitLearnBase

using PyCall
using Parameters
using SymDict

include("sk_utils.jl")

importall ScikitLearnBase

# These definitions are potentially costly. Get rid of the pywrap?
sklearn() = pywrap(pyimport("sklearn"))
sk_base() = pywrap(pyimport("sklearn.base"))

# This should be in ScikitLearn.jl, maybe?
translated_modules =
    Dict(:cross_validation => :CrossValidation,
         :pipeline => :Pipelines,
         :grid_search => :GridSearch)

for f in ScikitLearnBase.api
    # Export all the API. Not sure if we should do that...
    @eval(export $f)
end


# Note that I don't know the rationale for the `safe` argument - cstjean Feb2016
clone(py_model::PyObject) = sklearn().clone(py_model, safe=true)
is_classifier(py_model::PyObject) = sk_base().is_classifier(py_model)

is_pairwise(estimator) = false # global default - override for specific models
is_pairwise(py_model::PyObject) =
    haskey(py_model, "_pairwise") ? py_model[:_pairwise] : false

get_classes(py_estimator::PyObject) = py_estimator[:classes_]
get_components(py_estimator::PyObject) = py_estimator[:components_]
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
               :partial_fit! => :partial_fit,
               :score_samples => :score_samples,
               :sample => :sample,
               :score => :score,
               :transform => :transform,
               :set_params! => :set_params)

# PyCall does not always convert everything back into a Julia value,
# unfortunately, so we have some post-evaluation logic. These should be fixed
# in PyCall.jl
tweak_rval(x) = x
function tweak_rval(x::PyObject)
    @pyimport numpy
    if pyisinstance(x, numpy.ndarray) && length(x[:shape]) == 1
        return collect(x)
    else
        x
    end
end

for (jl_fun, py_fun) in api_map
    @eval $jl_fun(py_model::PyObject, args...; kwargs...) =
        tweak_rval(py_model[$(Expr(:quote, py_fun))](args...; kwargs...))
end

################################################################################

"""
@sk_import imports models from the Python version of scikit-learn. Example:

    @sk_import linear_model: (LinearRegression, LogisticRegression)
    model = fit!(LinearRegression(), X, y)
"""
macro sk_import(expr)
    @assert @capture(expr, mod_:what_) "`@sk_import` syntax error. Try something like: @sk_import linear_model: (LinearRegression, LogisticRegression)"
    if haskey(translated_modules, mod)
        warn("Module $mod has been ported to Julia - try `import ScikitLearn: $(translated_modules[mod])` instead")
    end
    :(Skcore.@pyimport2($(esc(Expr(:., :sklearn, mod))): $(esc(what))))
end

################################################################################

# A CompositeEstimator contains one or more estimators
abstract CompositeEstimator <: BaseEstimator

function set_params!(estimator::CompositeEstimator; params...) # from base.py
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
            set_params!(sub_object; Dict(Symbol(sub_name)=>value)...)
        else
            TODO() # should be straight-forward
        end
    end
    estimator
end

################################################################################

include("pipeline.jl")
include("scorer.jl")
include("cross_validation.jl")
include("grid_search.jl")


end
