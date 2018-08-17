# This is useful in part because it completely ignores X - so X can be a
# dataframe, an empty array, or anything.
mutable struct ConstantRegressor{T} <: BaseRegressor
    value::T
    (::Type{ConstantRegressor{T}}){T}() = new{T}()
end
ConstantRegressor() = ConstantRegressor{Float64}()
@declare_hyperparameters(ConstantRegressor, Symbol[])

function ScikitLearnBase.fit!(cr::ConstantRegressor, X, y)
    cr.value = mean(y)
    return cr
end

ScikitLearnBase.predict(cr::ConstantRegressor, X) = cr.value

################################################################################

mutable struct FixedConstant{T} <: BaseRegressor
    value::T
    (::Type{FixedConstant{T}}){T}(; value=0.0) = new{T}(value)
end
FixedConstant{T}(; value::T=0.0) = FixedConstant{T}(value=value)
@declare_hyperparameters(FixedConstant, Symbol[:value])

ScikitLearnBase.fit!(fc::FixedConstant, X, y) = fc
ScikitLearnBase.predict(fc::FixedConstant, X) = fc.value
