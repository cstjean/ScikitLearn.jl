# Skcore contains the actual implementation of the Julia part of scikit-learn.
# It's a flat module, which simplifies the implementation.
# Sklearn defines the visible interface. This schema makes it easier
# change the public module structure.
module Skcore

using Random, LinearAlgebra
using ScikitLearnBase
using SparseArrays

using PyCall
using Parameters
using Compat
import VersionParsing

for f in ScikitLearnBase.api
    # Used to be importall, but no longer exists in 0.7
    @eval import ScikitLearnBase: $f
end

include("init.jl")
include("sk_utils.jl")

""" Like `pyimport` but gives a more informative error message """
function importpy(name::AbstractString)
    try
        return pyimport(name)
    catch e
        if isa(e, PyCall.PyError)
            error("This ScikitLearn.jl functionality ($name) requires installing the Python scikit-learn library. See instructions on https://github.com/cstjean/ScikitLearn.jl")
        else
            rethrow()
        end
    end
end


# These definitions are potentially costly. Get rid of the pywrap?
sklearn() = importpy("sklearn")
sk_base() = importpy("sklearn.base")

# This should be in ScikitLearn.jl, maybe?
translated_modules =
    Dict(:model_selection => :CrossValidation,
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
    hasproperty(py_model, :_pairwise) ? py_model._pairwise : false

get_classes(py_estimator::PyObject) = py_estimator.classes_
get_components(py_estimator::PyObject) = py_estimator.components_

# Not the cleanest of definitions
is_transformer(estimator::Type) = !isempty(methods(transform, (estimator, Any)))
is_transformer(estimator::Any) = is_transformer(typeof(estimator))
################################################################################

# Julia => Python
api_map = Dict(:decision_function => :decision_function,
               :fit! => :fit,
               :fit_predict! => :fit_predict,
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
    numpy = importpy("numpy")
    if pyisinstance(x, numpy.ndarray) && length(x.shape) == 1
        return collect(x)
    else
        x
    end
end

for (jl_fun, py_fun) in api_map
    @eval $jl_fun(py_model::PyObject, args...; kwargs...) =
        tweak_rval(py_model.$(py_fun)(args...; kwargs...))
end

""" `predict_nc(model, X)` calls predict on the Python `model`, but returns
the result as a `PyArray`, which is more efficient than the usual path. See
PyCall.jl """
predict_nc(model::PyObject, X) = pycall(model.predict, PyArray, X)
predict_nc(model::Any, X) = predict(model, X) # default

################################################################################

symbols_in(e::Expr) = union(symbols_in(e.head), map(symbols_in, e.args)...)
symbols_in(e::Symbol) = Set([e])
symbols_in(::Any) = Set()



"""
@sk_import imports models from the Python version of scikit-learn. For instance, the
Julia equivalent of
`from sklearn.linear_model import LinearRegression, LogicisticRegression` is:
    @sk_import linear_model: (LinearRegression, LogisticRegression)
    model = fit!(LinearRegression(), X, y)
"""
macro sk_import(expr)
    @assert @capture(expr, mod_:what_) "`@sk_import` syntax error. Try something like: @sk_import linear_model: (LinearRegression, LogisticRegression)"
    if haskey(translated_modules, mod)
        @warn("Module $mod has been ported to Julia - try `import ScikitLearn: $(translated_modules[mod])` instead")
    end
    if :sklearn in symbols_in(expr)
        error("Bad @sk_import: please remove `sklearn.` (it is implicit)")
    end
    if isa(what, Symbol)
        members = [what]
    else
        @assert @capture(what, ((members__),)) "Bad @sk_import statement"
    end
    mod_string = "sklearn.$mod"
    :(begin
        # Make sure that sklearn is installed.
       # $Skcore.import_sklearn()
        # We used to rely on @pyimport2, but that macro unfortunately loads the Python
        # module at macro-expansion-time, which happens before Skcore.import_sklearn().
        # The new `pyimport`-based implementation is cleaner - Mar'17
        mod_obj = pyimport($mod_string)
        $([:(const $(esc(w)) = mod_obj.$(w))
           for w in members]...)
    end)
end

################################################################################

# A CompositeEstimator contains one or more estimators
@compat abstract type CompositeEstimator <: BaseEstimator end

function set_params!(estimator::CompositeEstimator; params...) # from base.py
    # Simple optimisation to gain speed (inspect is slow)
    if isempty(params) return estimator end

    valid_params = get_params(estimator, deep=true)
    for (key, value) in params
        sp = split(string(key), "__"; limit=2)
        if length(sp) > 1
            name, sub_name = sp
            if !haskey(valid_params, name::AbstractString)
                throw(ArgumentError("Invalid parameter ($name) for estimator $estimator"))
            end
            sub_object = valid_params[name]
            set_params!(sub_object; Dict(Symbol(sub_name)=>value)...)
        else
            # TODO: AFAICT this is supported by sklearn, but I'd have to check how it's
            # handled.
            error("Bad parameter name: $key. For a $(typeof(estimator)), parameter names should be of the form :componentname__feature")
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
