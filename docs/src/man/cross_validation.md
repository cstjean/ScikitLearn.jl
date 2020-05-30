Cross-validation
-----
```@meta
DocTestSetup = quote
    using ScikitLearn
using RDatasets: dataset
@sk_import linear_model: LogisticRegression
iris = dataset("datasets", "iris")
X = convert(Array, iris[!, [:SepalLength, :SepalWidth, :PetalLength,
        :PetalWidth]])
y = convert(Array, iris[!,:Species])
end
```
Cross-validation in ScikitLearn.jl is the same as in scikit-learn:
```@setup cross_val
using ScikitLearn
using RDatasets: dataset
@sk_import linear_model: LogisticRegression
iris = dataset("datasets", "iris")
X = convert(Array, iris[!, [:SepalLength, :SepalWidth, :PetalLength, 
        :PetalWidth]])
y = convert(Array, iris[!,:Species])

```

```@repl cross_val
using ScikitLearn.CrossValidation: cross_val_score

cross_val_score(LogisticRegression(max_iter=150), X, y; cv=5)  # 5-fold

```

See `?cross_val_score` and the [user guide](http://scikit-learn.org/stable/modules/cross_validation.html) for details.

We support all the [scikit-learn cross-validation iterators](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators) (KFold,
StratifiedKFold, etc.) For example:

```@jldoctest
julia> ScikitLearn.CrossValidation.KFold(10, n_folds=3)
3-element Array{Tuple{Array{Int64,1},Array{Int64,1}},1}:
 ([5, 6, 7, 8, 9, 10], [1, 2, 3, 4])
 ([1, 2, 3, 4, 8, 9, 10], [5, 6, 7])
 ([1, 2, 3, 4, 5, 6, 7], [8, 9, 10])

```

These iterators can be passed to `cross_val_score`'s `cv` argument.

Note: the most common iterators have been translated to Julia. The others still
require scikit-learn (python) to be installed.

### Examples

- [Quick start guide](quickstart.md)
- [Concatenating multiple feature extraction methods](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Feature_Stacker.ipynb)
- [Underfitting vs. Overfitting](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Underfitting_vs_Overfitting.ipynb)

## Cross-validated predictions

`cross_val_predict` performs cross-validation and returns the test-set predicted
values. Documentation [here](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_predict.html)

### Examples

- [Cross_Validated_Predictions](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Cross_Validated_Predictions.ipynb)
```@meta
DocTestSetup = nothing
```

