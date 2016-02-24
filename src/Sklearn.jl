module Sklearn

using PyCall

include("utils.jl")

# API
export fit!, transform, fit_transform!, predict, score_samples, sample, score
export decision_function

abstract BaseEstimator

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
type Pipeline <: BaseEstimator
    fitted::Bool
    steps::Vector # of tuples(string, model)
    function Pipeline(steps)
        @assert(nunique(map(first, steps)) == length(steps),
                "Pipeline's models' names must all be unique")
        new(false, steps)
    end
end

get_models(pip::Pipeline) = map(second, pip)
get_transforms(pip::Pipeline) = get_models(pip)[1:end-1]
get_estimator(pip::Pipeline) = get_models(pip)[end]

function fit!(pip::Pipeline, X0, y=None)
    X = X0
    for tr in get_transforms(pip)
        # sklearn passes the y target to the transforms, and I'm not sure
        # why, but some of the rationale is here:
        # https://github.com/scikit-learn/scikit-learn/pull/3113
        # We could support transforms on y fairly easily if we
        # used an interface with the default:
        #    transform_xy(model, X, y) = (transform(model, X), y)
        X = fit_transform!(tr, X, y)
    end
    est = get_estimator(pip)
    fit!(est, X, y)
    pip
end

################################################################################

api_map = Dict(:fit! => :fit,
               :predict => :predict,
               :decision_function => :decision_function,
               :fit_transform! => :fit_transform,
               :score_samples => :score_samples,
               :sample => :sample,
               :score => :score)

for (jl_fun, py_fun) in api_map
    @eval $jl_fun(py_model::PyObject, args...; kwargs...) =
        py_model[$(Expr(:quote, py_fun))](args...; kwargs...)
end


end
