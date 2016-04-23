using DecisionTree
using DecisionTree: Leaf, Node, LeafOrNode
importall ScikitLearnBase

################################################################################
# Helpers

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
class_index(classes) =
    Dict([c=>i for (i, c) in enumerate(classes)])

"""    compute_probabilities(classes::Vector, votes::Vector)

Helper function. Counts the votes.
Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
order as `get_classes(classifier)` (eg. ["versicolor", "setosa", "virginica"])
"""
function compute_probabilities(classes::Vector, votes::Vector)
    class2ind = class_index(classes)
    counts = zeros(Float64, length(class2ind))
    for label in votes
        counts[class2ind[label]] += 1
    end
    return counts / sum(counts) # normalize to get probabilities
end

################################################################################
# Classifier

type DecisionTreeClassifier <: BaseClassifier
    pruning_purity_threshold::Nullable{Float64} # no pruning if null
    nsubfeatures::Int
    root::Node
    classes::Vector   # an arbitrary ordering of the labels 
    # No pruning by default
    DecisionTreeClassifier(;pruning_purity_threshold=nothing, nsubfeatures=0) =
        new(convert(Nullable{Float64}, pruning_purity_threshold), nsubfeatures)
end

get_classes(dt::DecisionTreeClassifier) = dt.classes
declare_hyperparameters(DecisionTreeClassifier,
                        [:pruning_purity_threshold, :nsubfeatures])

function fit!(dt::DecisionTreeClassifier, X, y)
    dt.root = build_tree(y, X, dt.nsubfeatures)
    if !isnull(dt.pruning_purity_threshold)
        dt.root = prune_tree(dt.root, get(dt.pruning_purity_threshold))
    end
    dt.classes = sort(unique(y))
    dt
end

predict(dt::DecisionTreeClassifier, X) = apply_tree(dt.root, X)

# apply_tree_prob computes P(C=c|X) by counting the fraction of C objects
# in leaf.values
apply_tree_prob(leaf::Leaf, feature::Vector, classes) =
    compute_probabilities(classes, leaf.values)

function apply_tree_prob(tree::Node, features::Vector, classes)
    if tree.featval === nothing
        return apply_tree_prob(tree.left, features, classes)
    elseif features[tree.featid] < tree.featval
        return apply_tree_prob(tree.left, features, classes)
    else
        return apply_tree_prob(tree.right, features, classes)
    end
end

function apply_tree_prob(tree::LeafOrNode, features::Matrix, classes)
    N = size(features,1)
    predictions = Array(Float64, N, length(classes))
    for i in 1:N
        predictions[i, :] = apply_tree_prob(tree, squeeze(features[i,:],1),
                                            classes)
    end
    return predictions
end

predict_proba(dt::DecisionTreeClassifier, X) =
    apply_tree_prob(dt.root, X, dt.classes)

predict_log_proba(dt::DecisionTreeClassifier, X) =
    log(predict_proba(dt, X)) # this will yield -Inf when p=0. Hmmm...

################################################################################
# Regression

type DecisionTreeRegressor <: BaseRegressor
    pruning_purity_threshold::Nullable{Float64}
    nsubfeatures::Int
    root::Node
    # No pruning by default (I think purity_threshold=1.0 is a no-op, maybe
    # we could use that)
    DecisionTreeRegressor(;pruning_purity_threshold=nothing, nsubfeatures=0) =
        new(convert(Nullable{Float64}, pruning_purity_threshold), nsubfeatures)
end

declare_hyperparameters(DecisionTreeRegressor,
                        [:pruning_purity_threshold, :nsubfeatures])

function fit!{T<:Real}(dt::DecisionTreeRegressor, X::Matrix, y::Vector{T})
    # build_tree knows that its a regression problem by its argument types. I'm
    # not sure why X has to be Float64, but the method signature requires it
    # (as of April 2016).
    dt.root = build_tree(y, convert(Matrix{Float64}, X), dt.nsubfeatures)
    if !isnull(dt.pruning_purity_threshold)
        dt.root = prune_tree(dt.root, get(dt.pruning_purity_threshold))
    end
    dt
end

predict(dt::DecisionTreeRegressor, X) = apply_tree(dt.root, X)

################################################################################
# Random Forest Classification

type RandomForestClassifier <: BaseClassifier
    nsubfeatures::Int
    ntrees::Int
    partialsampling::Float64
    ensemble::Ensemble
    classes::Vector
    RandomForestClassifier(; nsubfeatures=0, ntrees=10, partialsampling=0.7) = 
        new(nsubfeatures, ntrees, partialsampling)
end

get_classes(dt::DecisionTreeClassifier) = dt.classes
declare_hyperparameters(DecisionTreeClassifier,
                        [:nsubfeatures, :ntrees, :partialsampling])

function fit!(rf::RandomForestClassifier, X::Matrix, y::Vector)
    rf.ensemble = build_forest(y, X, rf.nsubfeatures, rf.ntrees,
                               rf.partialsampling)
    rf.classes = sort(unique(y))
    rf
end

function apply_forest_prob(forest::Ensemble, features::Vector, classes)
    votes = [apply_tree(tree, features) for tree in forest.trees]
    return compute_probabilities(classes, votes)
end

function apply_forest_prob(forest::Ensemble, features::Matrix, classes)
    N = size(features,1)
    predictions = Array(Float64, N, length(classes))
    for i in 1:N
        predictions[i, :] = apply_forest_prob(forest, squeeze(features[i,:],1),
                                              classes)
    end
    return predictions
end

predict_proba(rf::RandomForestClassifier, X) = 
    apply_forest_prob(rf.ensemble, X, rf.classes)

predict(rf::RandomForestClassifier, X) = apply_forest(rf.ensemble, X)

################################################################################
# Random Forest Regression

type RandomForestRegressor <: BaseRegressor
    nsubfeatures::Int
    ntrees::Int
    partialsampling::Float64
    ensemble::Ensemble
    RandomForestRegressor(; nsubfeatures=0, ntrees=10, partialsampling=0.7) = 
        new(nsubfeatures, ntrees, partialsampling)
end

declare_hyperparameters(DecisionTreeRegressor,
                        [:nsubfeatures, :ntrees, :partialsampling])

function fit!{T<:Real}(rf::RandomForestRegressor, X::Matrix, y::Vector{T})
    rf.ensemble = build_forest(y, convert(Matrix{Float64}, X), rf.nsubfeatures,
                               rf.ntrees, rf.partialsampling)
    rf
end

predict(rf::RandomForestRegressor, X) = apply_forest(rf.ensemble, X)

