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
    steps::Vector # of tuples(string, model)
    function Pipeline(steps)
        @assert(nunique(map(first, steps)) == length(steps),
                "Pipeline's models' names must all be unique")
        new(steps)
    end
end

get_models(pip::Pipeline) = map(second, pip.steps)
get_transforms(pip::Pipeline) = get_models(pip)[1:end-1]
get_estimator(pip::Pipeline) = get_models(pip)[end]

is_classifier(pip::Pipeline) = is_classifier(get_estimator(pip))
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
    est = get_estimator(pip)
    fit!(est, X, y)
    pip
end

function predict(pip::Pipeline, X)
    Xt = X
    for transf in get_transforms(pip)
        Xt = transform(transf, Xt)
    end
    return predict(get_estimator(pip), Xt)
end
