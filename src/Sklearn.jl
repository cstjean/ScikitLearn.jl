module Sklearn

using PyCall
using Utils

include("sk_utils.jl")

# API
export fit!, transform, fit_transform!, predict, score_samples, sample, score
export decision_function

abstract BaseEstimator

include("pipeline.jl")


################################################################################

api_map = Dict(:fit! => :fit,
               :predict => :predict,
               :decision_function => :decision_function,
               :fit_transform! => :fit_transform,
               :transform => :transform, 
               :score_samples => :score_samples,
               :sample => :sample,
               :score => :score)

for (jl_fun, py_fun) in api_map
    @eval $jl_fun(py_model::PyObject, args...; kwargs...) =
        py_model[$(Expr(:quote, py_fun))](args...; kwargs...)
end


end
