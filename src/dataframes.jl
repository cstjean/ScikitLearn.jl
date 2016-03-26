# Python Author: Paul Butler (https://github.com/paulgb/sklearn-pandas)
# Julia adaptation: Cedric St-Jean

export DataFrameMapper

importall ScikitLearnBase

using DataFrames: DataFrame

_build_transformer(transformers) = transformers

function _build_transformer(transformers::Vector)
    make_pipeline(transformers...)
end

"""
Map DataFrame column subsets to their own scikit-learn transformation.

Params:
features    a vector of pairs. The first element is the column name.
            This can be a symbol (for one column) or a list
            of symbols. The second element is an object that supports
            sklearn's transform interface, or a list of such objects.
sparse      will return sparse matrix if set to true and any of the
            extracted features is sparse. Defaults to False.
"""
type DataFrameMapper <: BaseEstimator
    features::Vector{Tuple}
    sparse
    function DataFrameMapper(features, sparse=false)
        @assert !sparse "TODO: support sparse"
        features = [(columns, _build_transformer(transformers))
                    for (columns, transformers) in features]
        new(features, sparse)
    end
end

"""
Convert a vector to a matrix
"""
_handle_feature(fea::Vector) = reshape(fea, length(fea), 1)
_handle_feature(fea::Matrix) = fea

"""
Get a subset of columns from the given table X.
X       a dataframe; the table to select columns from
cols    a string or list of strings representing the columns
        to select
Returns a matrix with the data from the selected columns
"""
function _get_col_subset(X, cols; return_vector=false)
    if isa(X, Vector)
        X = [x[cols] for x in X]
        X = DataFrame(X)
    # Julia note: not sure what role DataWrapper serves, ignoring it for now
    ## elseif isinstance(X, DataWrapper)
    ##     # if it's a datawrapper, unwrap it
    ##     X = X.df
    end

    # Maybe we could keep it as a DataArray, but I'd rather be conservative
    # for now.
    return convert(Array, return_vector ? X[cols[1]] : X[cols])
end

_get_col_subset(X, col::Symbol) =
    _get_col_subset(X, [col], return_vector=true)


function fit!(self::DataFrameMapper, X, y=nothing; kwargs...)
    for (columns, transformers) in self.features
        if transformers !== nothing
            fit!(transformers, _get_col_subset(X, columns))
        end
    end
    return self
end

function transform(self, X)
    extracted = []
    for (columns, transformers) in self.features
        # columns could be a string or list of strings; we don't care because
        # DataFrame indexing handles both
        Xt = _get_col_subset(X, columns)
        if transformers !== nothing
            Xt = transform(transformers, Xt)
        end
        push!(extracted, _handle_feature(Xt))
    end

    # combine the feature outputs into one array. At this point we lose track
    # of which features were created from which input columns, so it's assumed
    # that that doesn't matter to the model.

    # If any of the extracted features is sparse, combine sparsely.
    # Otherwise, combine as normal arrays.
    # Julia TODO
    ## if any(sparse.issparse(fea) for fea in extracted):
    ##     stacked = sparse.hstack(extracted).tocsr()
    ##     # return a sparse matrix only if the mapper was initialized
    ##     # with sparse=True
    ##     if not self.sparse:
    ##         return stacked.toarray()
    ## else:

    return hcat(extracted...)
end

