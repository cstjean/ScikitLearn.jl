# The quick-start guide code. Some of these are repro tests

using Test
using RDatasets: dataset

iris = dataset("datasets", "iris")

X = Array(iris[!, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = iris[:, :Species]

using ScikitLearn

@sk_import linear_model: LogisticRegression

model = LogisticRegression(max_iter=200, random_state=20)

fit!(model, X, y)

accuracy = sum(predict(model, X) .== y) / length(y)
@test accuracy ≈ 0.97333333333

using ScikitLearn.CrossValidation: cross_val_score

@test cross_val_score(LogisticRegression(random_state=25, max_iter=200), X, y; cv=5) ≈ [0.96666667, 1.0, 0.93333333, 0.96666667, 1.0] atol=0.0001
