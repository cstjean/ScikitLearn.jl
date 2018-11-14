using Compat: @compat

mutable struct LinearRegression{T <: Array} <: BaseRegressor
    coefs::T
    intercept::Real
    LinearRegression{T}() where T = new{T}()
end

"""    LinearRegression(; eltype=Float64, multi_output=nothing)

Linear regression. Supports both single-output and multiple-output regression.
Optimized for speed.

- `eltype`: the element type of the coefficient array. `Float64` is generally
best for numerical reasons.
- `multi_output`: for maximum efficiency, specify `multi_output=true/false` """
function LinearRegression(; eltype=Float64, multi_output=nothing)
    if multi_output === nothing
        LinearRegression{Array{eltype}}()
    elseif multi_output::Bool
        LinearRegression{Array{eltype, 2}}()
    else
        LinearRegression{Array{eltype, 1}}()
    end
end

@declare_hyperparameters(LinearRegression, Symbol[])

function ScikitLearnBase.fit!(lr::LinearRegression, X::AbstractArray{XT},
                              y::AbstractArray{yT}) where {XT, yT}
    if XT == Float32 || yT == Float32
        warn("Regression on Float32 is prone to inaccuracy")
    end
    results = [ones(size(X, 2), 1) X'] \ y
    lr.intercept = results[1];
    lr.coefs = results[2:end];
end

ScikitLearnBase.predict(lr::LinearRegression, X) = lr.coefs' * X .+ lr.intercept
