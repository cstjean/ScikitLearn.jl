using ScikitLearn
using Test

@test predict(fit!(clone(ScikitLearn.Models.FixedFunctionRegressor(predict_fn=sum)), Matrix(1.0I, 10, 10), [1,2,3]), ones(5, 2)) == [2.0, 2.0, 2.0, 2.0, 2.0]
