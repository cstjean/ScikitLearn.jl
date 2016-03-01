Relationship to scikit-learn
------

The [scikit-learn](http://scikit-learn.org/stable/about.html) library has so far received [contributions](https://github.com/scikit-learn/scikit-learn/graphs/contributors) from hundreds of developers and machine learning experts. ScikitLearn.jl leans on that codebase as much as possible, but the API had to be adapated to Julia, and follows Julia's conventions.

It is important to understand those changes in order to read the scikit-learn [documentation](http://scikit-learn.org/stable/documentation.html). In particular:

- Methods are called the usual way: Python's `model.predict(X)` becomes
  `predict(model, X)`)
- Methods that modify the model's state have a `!` at the end:
  `model.fit_transform(X)` becomes `fit_transform!(model, X)`
- Submodules follow the camel case and pluralizing (where appropriate)
  conventions: `sklearn.cross_validation` becomes `ScikitLearn.CrossValidation`
  and `sklearn.pipelines.Pipeline` becomes `ScikitLearn.Pipelines.Pipeline`

When training a scikit-learn model, you can access its attributes with `[]`. For
example:

```
@sk_import linear_model: Lasso
lm = fit!(Lasso(), X, y)
println(lm[:n_iter_])   # equivalent to lm.n_iter_ in Python
```