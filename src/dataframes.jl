# Python Author: Paul Butler (https://github.com/paulgb/sklearn-pandas)
# Julia adaptation: Cedric St-Jean

export DataFrameMapper, DataFrameColSelector

using DataFrames: DataFrame, ismissing, eachcol,
                  AbstractDataFrame, DataFrameRow
using SparseArrays

_build_transformer(transformers) = transformers

function _build_transformer(transformers::Vector)
    Pipelines.make_pipeline(transformers...)
end

"""
    DataFrameMapper(features; sparse=false, missing2NaN=false,
                    output_type=Matrix{Float64})

Map DataFrame column subsets to their own scikit-learn transformation.

Arguments:

- **features**:    a vector of pairs. The first element is the column name.
                   This can be a symbol (for one column) or a vector
                   of symbols. The second element is an object that supports
                   sklearn's transform interface, or a vector of such objects.
- **sparse**       will return sparse matrix if set to true and any of the
                   extracted features is sparse. Defaults to False.
- **missing2NaN**  will convert missing values to NaNs (necessary for Python
                   models)
- **output_type**: the type of the result (default: Matrix{Float64})
"""
mutable struct DataFrameMapper <: BaseEstimator
    features::Vector{Tuple}
    sparse::Bool
    missing2NaN::Bool
    output_type::Type
    function DataFrameMapper(features; sparse=false, missing2NaN=false,
                             # In scikit-learn, accepting 1D arguments is
                             # deprecated. See scikit-learn/pull/4511
                             output_type=Matrix{Float64})
        # Input validation
        for (col_name, transformer) in features
            @assert(isa(col_name, Union{Symbol, Vector}),
                    "Bad DataFrameMapper features, see docstring")
            @assert(isa(transformer, Union{Nothing, Vector}) ||
                    ScikitLearn.Utils.is_transformer(transformer),
                    "Bad DataFrameMapper features, see docstring")
        end
        @assert !sparse "TODO: support sparse"
        features = Tuple[(columns, _build_transformer(transformers))
                         for (columns, transformers) in features]
        new(features, sparse, missing2NaN, output_type)
    end
end

ScikitLearnBase.clone(dfm::DataFrameMapper) =
    DataFrameMapper([(col, feat===nothing ? feat : clone(feat))
                     for (col, feat) in dfm.features];
                    sparse=dfm.sparse, missing2NaN=dfm.missing2NaN)
    

"""
Convert a vector to a matrix
"""
_handle_feature(fea::Vector) = reshape(fea, length(fea), 1)
_handle_feature(fea::Matrix) = fea

"""
Get a subset of columns from the given table X.
X       a dataframe; the table to select columns from
cols    a symbol or vector of symbols representing the columns
        to select
Returns a matrix with the data from the selected columns
"""
function _get_col_subset(X, cols::Vector{Symbol}; return_vector=false)
    if isa(X, Vector)
        TODO() # I'm not sure that this pathway has ever been tried. Not even
               # sure what it's trying to accomplish! - @cstjean
        X = [x[cols] for x in X]
        X = DataFrame(X)
    # Julia note: not sure what role DataWrapper serves, ignoring it for now
    ## elseif isinstance(X, DataWrapper)
    ##     # if it's a datawrapper, unwrap it
    ##     X = X.df
    end

    try
        return Array(return_vector ? X[:, cols[1]] : X[:, cols])
    catch e
        if isa(e, KeyError)
            throw(KeyError("DataFrameMapper: in dataframe (size=$(size(X)), cols=$(names(X))), $cols")) # not found
        else rethrow() end
    end
end

_get_col_subset(X, col::Symbol) =
    _get_col_subset(X, [col], return_vector=true)

function _maybe_convert_missing(dfm::DataFrameMapper, X::DataFrame)
    # The type to promote to (must be able to contain NaN)
    sup_type(::Type{<:Number}) = Float64
    sup_type(::Type) = Any
    if dfm.missing2NaN
        X = deepcopy(X)
        # There might be a much simpler way of doing this with DataFrames
        for col in names(X)
            values = X[!, col]
            na_inds = ismissing.(values)
            # We can't put a NaN in an array of Int. This is ugly code, FIXME
            if any(na_inds)
                if !(eltype(values) <: AbstractFloat)
                    values = convert(Vector{sup_type(eltype(values))},
                                     copy(values))
                else
                    values = copy(values) # since we'll modify it
                end
                values[na_inds] .= NaN
                X[!, col] .= values
            end
        end
    end
    return X
end

function ScikitLearnBase.fit!(self::DataFrameMapper, X, y=nothing; kwargs...)
    X = _maybe_convert_missing(self, X)
    for (columns, transformers) in self.features
        if transformers !== nothing
            fit!(transformers, _get_col_subset(X, columns))
        end
    end
    return self
end

function ScikitLearnBase.transform(self::DataFrameMapper, X::DataFrame)
    X = _maybe_convert_missing(self, X)
    extracted = []
    for (columns, transformers) in self.features
        # columns could be a string or list of strings; we don't care because
        # DataFrame indexing handles both
        Xt = _get_col_subset(X, columns)
        if transformers !== nothing
            Xt = self.output_type(transform(transformers, Xt))
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

ScikitLearnBase.transform(dfm::DataFrameMapper, X::Vector{T}) where {T<:Dict} =
    # This could be handled better...
    transform(dfm, DataFrame(X))
ScikitLearnBase.transform(dfm::DataFrameMapper, X::Vector{T}) where {T<:DataFrameRow} =
    # This could be handled much, much better...
    transform(dfm, [Dict(dfr) for dfr in X])

SparseArrays.issparse(::DataFrame) = false

################################################################################

"""    DataFrameColSelector(colums::Vector{Symbol}; output_type=Matrix{Float64}

This is a pared-down, less featureful version of `DataFrameMapper`, but it
also runs faster. It only allows selecting columns from the input `DataFrame`.
Use in a pipeline. """
struct DataFrameColSelector <: BaseEstimator
    cols::Vector{Symbol}
    output_type::Type
    DataFrameColSelector(cols; output_type=Matrix{Float64}) = 
        new(cols, output_type)
end

ScikitLearnBase.clone(dfcs::DataFrameColSelector) =
    DataFrameColSelector(dfcs.cols; output_type=dfcs.output_type)
ScikitLearnBase.fit!(dfcs::DataFrameColSelector, X, y=nothing) = dfcs
ScikitLearnBase.transform(dfcs::DataFrameColSelector, X::AbstractDataFrame) =
    dfcs.output_type(X[:, dfcs.cols])

ScikitLearnBase.transform(dfcs::DataFrameColSelector, X) =
    _transform(dfcs, X, dfcs.output_type)
_transform(dfcs::DataFrameColSelector, X::Vector{T},
           ::Type{Matrix{O}}) where {T<:DataFrameRow, O} =
    O[dfr[col] for dfr in X, col in dfcs.cols]

################################################################################

# This is an unexported hack for my own personal use - cstjean
function dummy_input(dfm::DataFrameMapper)
    rng = MersenneTwister(11)
    N = sum([length(cols) for (cols, _) in dfm.features])
    df = DataFrame()
    for (cols, _) in dfm.features
        for col in cols
            df[!, col] = rand(rng, N)
        end
    end
    df
end
