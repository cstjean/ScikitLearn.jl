module Sklearn

using PyCall
using Utils

include("sk_utils.jl")

@pyimport2 sklearn
@pyimport sklearn.base as sk_base

# API
export fit!, transform, fit_transform!, predict, score_samples, sample, score
export decision_function, score, clone, is_classifier 

abstract BaseEstimator

include("pipeline.jl")
include("cross_validation.jl")

# Note that I don't know the rationale for the `safe` argument - cstjean Feb2016
clone(py_model::PyObject) = sklearn.clone(py_model, safe=true)
is_classifier(py_model::PyObject) = sk_base.is_classifier(py_model)

################################################################################

api_map = Dict(:decision_function => :decision_function,
               :fit! => :fit,
               :fit_transform! => :fit_transform,
               :predict => :predict,
               :score_samples => :score_samples,
               :sample => :sample,
               :score => :score,
               :transform => :transform)

for (jl_fun, py_fun) in api_map
    @eval $jl_fun(py_model::PyObject, args...; kwargs...) =
        py_model[$(Expr(:quote, py_fun))](args...; kwargs...)
end


end
