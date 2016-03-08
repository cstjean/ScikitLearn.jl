Cross-validation
-----

Cross-validation in ScikitLearn.jl is the same as in scikit-learn:

```julia
using ScikitLearn.CrossValidation: cross_val_score

cross_val_score(LogisticRegression(), X, y; cv=5)  # 5-fold
```

See `?cross_val_score` and the [user guide](http://scikit-learn.org/stable/modules/cross_validation.html) for details.

We support all the [scikit-learn cross-validation iterators](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators) (KFold,
StratifiedKFold, etc.) For example:

```julia
> ScikitLearn.CrossValidation.KFold(10, 3)

3-element Array{Tuple{Array{Int64,1},Array{Int64,1}},1}:
 ([5,6,7,8,9,10],[1,2,3,4])
 ([1,2,3,4,8,9,10],[5,6,7])
 ([1,2,3,4,5,6,7],[8,9,10])
```

These iterators can be passed to `cross_val_score`'s `cv` argument.

### Examples

- [Quick start guide](quickstart.md)
- [Concatenating multiple feature extraction methods](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Feature_Stacker.ipynb)
- [Underfitting vs. Overfitting](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Underfitting_vs_Overfitting.ipynb)