export DictEncoder

"""    DictEncoder()

For every unique row in the input matrix, associate a 0/1 binary column in the
output matrix. This is similar to OneHotEncoder, but considers the entire row
as a single value for one-hot-encoding. It works with any hashable datatype.

It is common to use it inside a DataFrameMapper, with a particular subset of
the columns.
"""
mutable struct DictEncoder <: BaseEstimator
    di::Dict{Any, Int}
    DictEncoder() = new(Dict{Any, Int}())
end

@declare_hyperparameters(DictEncoder, Symbol[])

function fit!(de::DictEncoder, X::AbstractMatrix, y=nothing)
    count = 1
    for row_no in 1:size(X, 1)
        # collect because of
        # https://github.com/JuliaStats/DataArrays.jl/issues/147
        row = collect(X[row_no, :])
        if !haskey(de.di, row)
            de.di[row] = count
            count += 1
        end
    end
    de
end


function transform(de::DictEncoder, X::AbstractMatrix)
    out = zeros(size(X, 1), length(de.di))
    for row_no in 1:size(X, 1)
        row = collect(X[row_no, :]) # see collect comment above
        if haskey(de.di, row )
            ind = de.di[row]
            out[row_no, ind] = 1.0
        end
    end
    out
end
