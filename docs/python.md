# Relationship to scikit-learn

The [scikit-learn](http://scikit-learn.org/stable/about.html) Python library has so far received [contributions](https://github.com/scikit-learn/scikit-learn/graphs/contributors) from dozens of developers and machine learning experts. ScikitLearn.jl leans on that codebase as much as possible, but the API had to be adapted to Julia, and follows Julia's conventions. When reading the scikit-learn [documentation](http://scikit-learn.org/stable/documentation.html), keep in mind that:

- Object methods are now functions: Python's `model.predict(X)` becomes
  `predict(model, X)`
- Methods that modify the model's state have a `!` at the end:
  `model.fit_transform(X)` becomes `fit_transform!(model, X)`
- Submodules that were translated into Julia follow the camel case and
  pluralizing (where appropriate) conventions: `sklearn.cross_validation`
  becomes `ScikitLearn.CrossValidation`, and `sklearn.pipelines.Pipeline`
  becomes `ScikitLearn.Pipelines.Pipeline`

To access the class members and methods of a Python objects (i.e. all
models imported through `@sk_import`), use `obj[:member_name]`. For example:

```
@sk_import linear_model: Lasso
lm = fit!(Lasso(), X, y)
println(lm[:n_iter_])   # equivalent to lm.n_iter_ in Python
```

