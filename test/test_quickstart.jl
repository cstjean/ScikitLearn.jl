# The quick-start guide code. Some of these are repro tests

using Test
using RDatasets: dataset

iris = dataset("datasets", "iris")

X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])

using ScikitLearn

@sk_import linear_model: LogisticRegression

model = LogisticRegression(fit_intercept=true, random_state=20)

fit!(model, X, y)

accuracy = sum(predict(model, X) .== y) / length(y)
@test accuracy == 0.96

using ScikitLearn.CrossValidation: cross_val_score

@test cross_val_score(LogisticRegression(random_state=25), X, y; cv=5) ≈ [1.0, 0.96666666667, 0.9333333333, 0.9, 1.0] atol=0.0001
