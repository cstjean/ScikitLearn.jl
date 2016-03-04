Python models
-----

scikit-learn has on the order of 100 to 200 models, split into three categories:

- [Supervised Learning](http://scikit-learn.org/stable/supervised_learning.html) (linear regression, support vector machines, random forest, neural nets, ...)
- [Unsupervised Learning](http://scikit-learn.org/stable/unsupervised_learning.html) (clustering, PCA, mixture models, manifold learning, ...)
- [Dataset Transformation](http://scikit-learn.org/stable/data_transforms.html)

Nearly all of those models will work with ScikitLearn.jl,
except for models that contain other models. Those had to be translated
into Julia. See [ScikitLearn.Pipelines](pipelines.md) for details.

To import a Python model, use `@sk_import`. For example, here's how to import
and fit `sklearn.linear_regression.LogisticRegression`

```
using ScikitLearn
@sk_import linear_model: LogisticRegression

log_reg = fit!(LogisticRegression(penalty="l1"), X_train, y_train)
predict(X_test)
```

Reminder: `?LogisticRegression` contains a lot of information about the model
parameters.



Julia models
------

To make your Julia model compatible with ScikitLearn.jl, you need to implement
the scikit-learn interface. See [ScikitLearnBase.jl](https://github.com/cstjean/ScikitLearnBase.jl)

