# This is useful in part because it completely ignores X - so X can be a
# dataframe, an empty array, whatever.
type ConstantRegressor{T} <: BaseRegressor
    value::T
    ConstantRegressor() = new()
end
ConstantRegressor() = ConstantRegressor{Float64}()
@declare_hyperparameters(ConstantRegressor, Symbol[])

function fit!(cr::ConstantRegressor, X, y)
    cr.value = mean(y)
    return cr
end

predict(cr::ConstantRegressor, X) = cr.value

################################################################################

type FixedConstant{T} <: BaseRegressor
    value::T
    FixedConstant(; value=0.0) = new(value)
end
FixedConstant{T}(; value::T=0.0) = FixedConstant{T}(value=value)
@declare_hyperparameters(FixedConstant, Symbol[:value])

fit!(fc::FixedConstant, X, y) = fc
predict(fc::FixedConstant, X) = fc.value
