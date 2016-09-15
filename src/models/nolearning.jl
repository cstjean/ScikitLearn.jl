""" `FixedFunctionRegressor(predict_fn::Function=row->...)` is a regression
model specified entirely by `fn`, for testing
purposes. `fit!(::FixedFunctionRegressor, ...)` is a no-op.

- `predict_fn(row)` is a function of one argument, which will be called on each
  row of the input matrix `X` that is passed to `predict(::FixedFunctionRegressor, X)`. """
type FixedFunctionRegressor <: BaseRegressor
    predict_fn::Function
    FixedFunctionRegressor(; predict_fn=error("Must provide `predict_fn`")) =
        new(predict_fn)
end

@declare_hyperparameters(FixedFunctionRegressor, Symbol[:predict_fn])

fit!(ffr::FixedFunctionRegressor, X, y) = ffr
predict(ffr::FixedFunctionRegressor, X) =
    squeeze(mapslices(ffr.predict_fn, X, 2), 2)
