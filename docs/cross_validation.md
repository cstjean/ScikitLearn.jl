Cross-validation
-----

Cross-validation in ScikitLearn.jl is the same as in scikit-learn:

```julia
using ScikitLearn.CrossValidation: cross_val_score

cross_val_score(LogisticRegression(), X, y; cv=5)  # 5-fold
```

See `?cross_val_score` and the [user guide](http://scikit-learn.org/stable/modules/cross_validation.html) for details.

### Examples

- [Quick start guide](quickstart.md)
- [Concatenating multiple feature extraction methods](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Feature_Stacker.ipynb)
- [Underfitting vs. Overfitting](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Underfitting_vs_Overfitting.ipynb)