using DecisionTree
importall ScikitLearnBase

type DecisionTreeClassifier <: BaseEstimator
    pruning_purity_threshold::Nullable{Float64}
    nsubfeatures::Int
    root::Node
    # No pruning by default
    DecisionTreeClassifier(;pruning_purity_threshold=nothing, nsubfeatures=0) =
        new(convert(Nullable{Float64}, pruning_purity_threshold), nsubfeatures)
end
# Same definition. Maybe it could be shared somehow? 
type DecisionTreeRegressor <: BaseEstimator
    pruning_purity_threshold::Nullable{Float64}
    nsubfeatures::Int
    root::Node
    # No pruning by default (I think purity_threshold=1.0 is a no-op, maybe
    # we could use that
    DecisionTreeRegressor(;pruning_purity_threshold=nothing, nsubfeatures=0) =
        new(convert(Nullable{Float64}, pruning_purity_threshold), nsubfeatures)
end

declare_hyperparameters(DecisionTreeClassifier,
                        [:pruning_purity_threshold, :nsubfeatures])
declare_hyperparameters(DecisionTreeRegressor,
                        [:pruning_purity_threshold, :nsubfeatures])

function fit!(dtc::DecisionTreeClassifier, X, y)
    dtc.root = build_tree(y, X, dtc.nsubfeatures)
    if !isnull(dtc.pruning_purity_threshold)
        dtc.root = prune_tree(dtc.root, get(dtc.pruning_purity_threshold))
    end
    dtc
end

function fit!{T<:Real}(dtc::DecisionTreeRegressor, X::Matrix, y::Vector{T})
    # build_tree knows that its a regression problem by its argument types. I'm
    # not sure why X has to be Float64, but the method signature requires it
    # (as of April 2016).
    dtc.root = build_tree(y, convert(Matrix{Float64}, X), dtc.nsubfeatures)
    if !isnull(dtc.pruning_purity_threshold)
        dtc.root = prune_tree(dtc.root, get(dtc.pruning_purity_threshold))
    end
    dtc
end

predict(dtc::Union{DecisionTreeClassifier, DecisionTreeRegressor}, X) =
    apply_tree(dtc.root, X)
