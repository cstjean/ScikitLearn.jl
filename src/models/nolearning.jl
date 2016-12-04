import ..Utils: isfit  # should isfit be moved to ScikitLearnBase?

""" `FixedFunctionRegressor(predict_fn::Function=row->...)` is a regression
model specified entirely by `fn`, for testing
purposes. `fit!(::FixedFunctionRegressor, ...)` is a no-op.

- `predict_fn(row)` is a function of one argument, which will be called on each
  row of the input matrix `X` that is passed to `predict(::FixedFunctionRegressor, X)`.
"""
type FixedFunctionRegressor <: BaseRegressor
    predict_fn::Function
    FixedFunctionRegressor(; predict_fn=error("Must provide `predict_fn`")) =
        new(predict_fn)
end

@declare_hyperparameters(FixedFunctionRegressor, Symbol[:predict_fn])

fit!(ffr::FixedFunctionRegressor, X, y) = ffr
predict(ffr::FixedFunctionRegressor, X) =
    squeeze(mapslices(ffr.predict_fn, X, 2), 2)

isfit(ffr::FixedFunctionRegressor) = true

################################################################################

""" `FixedFunctionClassifier(predict_fn::Function=row->..., predict_dist_fn=row->...)`
is a classifier model specified entirely by `fn`, for testing
purposes. `fit!(::FixedFunctionClassifier, ...)` is a no-op.

- `predict_fn(row)` is a function of one argument, which will be called on each
  row of the input matrix `X` that is passed to `predict(::FixedFunctionClassifier, X)`.
"""
type FixedFunctionClassifier <: BaseClassifier
    predict_fn::Function
    predict_dist_fn::Function
    FixedFunctionClassifier(; predict_fn=row->error("`predict_fn` not provided"),
                            predict_dist_fn=row->error("predict_dist_fn not provided")) =
        new(predict_fn, predict_dist_fn)
end


@declare_hyperparameters(FixedFunctionClassifier, Symbol[:predict_fn, :predict_dist_fn])

fit!(ffc::FixedFunctionClassifier, X, y) = ffc
predict(ffc::FixedFunctionClassifier, X) =
    squeeze(mapslices(ffc.predict_fn, X, 2), 2)
predict_dist(ffc::FixedFunctionClassifier, X) =
    squeeze(mapslices(ffc.predict_dist_fn, X, 2), 2)

isfit(ffc::FixedFunctionClassifier) = true
