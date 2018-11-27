using Compat: @compat

mutable struct LinearRegression{T<:Array,Y<:Number} <: BaseRegressor
    coefs::T
    intercepts::Array{Y, 1}
    LinearRegression{T,Y}() where{T,Y} = new{T, Y}()
end

"""    LinearRegression(; eltype=Float64, multi_output=nothing)

Linear regression. Supports both single-output and multiple-output regression.
Optimized for speed.

- `eltype`: the element type of the coefficient array. `Float64` is generally
best for numerical reasons.
- `multi_output`: for maximum efficiency, specify `multi_output=true/false` """
function LinearRegression(; eltype=Float64, multi_output::Union{Nothing, Bool}=nothing)
    if multi_output === nothing
        LinearRegression{Array{eltype}, eltype}()
    else
        LinearRegression{Array{eltype, 2}, eltype}()
    end
end

@declare_hyperparameters(LinearRegression, Symbol[])

function ScikitLearnBase.fit!(lr::LinearRegression, X::AbstractArray{XT},
                              y::AbstractArray{yT}) where {XT, yT}
    if XT == Float32 || yT == Float32
        warn("Regression on Float32 is prone to inaccuracy")
    end
    results = [ones(size(X, 2), 1) X'] \ y'
    lr.intercepts = results[1,:];
    lr.coefs = results[2:end,:];
    lr
end

ScikitLearnBase.predict(lr::LinearRegression, X) = lr.coefs' * X .+ lr.intercepts
