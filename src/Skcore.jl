# Skcore contains the actual implementation of the Julia part of scikit-learn.
# It's a flat module, which simplifies the implementation.
# Sklearn defines the visible interface. This schema makes it easier
# change the public module structure.
module Skcore

using SklearnBase
using SklearnBase: @import_api, @simple_estimator_constructor

using PyCall
using Parameters
using SymDict

include("sk_utils.jl")

importall SklearnBase

@pyimport2 sklearn
@pyimport sklearn.base as sk_base

for f in SklearnBase.api
    # Export all the API. Not sure if we should do that...
    @eval(export $f)
end


# Note that I don't know the rationale for the `safe` argument - cstjean Feb2016
clone(py_model::PyObject) = sklearn.clone(py_model, safe=true)
is_classifier(py_model::PyObject) = sk_base.is_classifier(py_model)

is_pairwise(estimator) = false # global default - override for specific models
is_pairwise(py_model::PyObject) =
    haskey(py_model, "_pairwise") ? py_model[:_pairwise] : false

get_classes(py_estimator::PyObject) = py_estimator[:classes_]
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
