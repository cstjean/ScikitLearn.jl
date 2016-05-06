Python models
-----

scikit-learn has on the order of 100 to 200 models (more generally called
"estimators"), split into three categories:

- [Supervised Learning](http://scikit-learn.org/stable/supervised_learning.html) (linear regression, support vector machines, random forest, neural nets, ...)
- [Unsupervised Learning](http://scikit-learn.org/stable/unsupervised_learning.html) (clustering, PCA, mixture models, manifold learning, ...)
- [Dataset Transformation](http://scikit-learn.org/stable/data_transforms.html) (preprocessing, text feature extraction, one-hot encoding, ...)

Nearly all of those estimators will work with ScikitLearn.jl,
except for estimators that contain other estimators. Those had to be translated
into Julia. See [ScikitLearn.Pipelines](pipelines.md) for details.

Python estimators are imported with `@sk_import`. For example, here's how to import
and fit `sklearn.linear_regression.LogisticRegression`

```julia
using ScikitLearn
@sk_import linear_model: LogisticRegression

log_reg = fit!(LogisticRegression(penalty="l1"), X_train, y_train)
predict(X_test)
```

Reminder: `?LogisticRegression` contains a lot of information about the model
parameters.



Julia models
------

Julia models are hosted in other packages, and need to be installed separately
with `Pkg.add` or `Pkg.checkout`. They all implement the [common api](api.md).

### GaussianMixtures.jl

```julia
Pkg.checkout("GaussianMixtures.jl")   # install the package
using GaussianMixtures: GMM
using ScikitLearn

gmm = fit!(GMM(n_components=3, # number of Gaussians to fit
               kind=:diag), # diagonal covariance matrix (other option: :full)
           X)
predict_proba(gmm, X)
```

Documentation at [GaussianMixtures.jl](https://github.com/davidavdav/GaussianMixtures.jl). See also this [density estimation example](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Density_Estimation_Julia.ipynb)

### GaussianProcesses.jl

```julia
Pkg.checkout("GaussianProcesses.jl")   # install the package
using GaussianProcesses: GP
using ScikitLearn

gp = fit!(GP(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-1e8),
          X, y)
predict(gp, X)
```

Documentation at [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl) and in the `?GP` docstring. [Example notebook](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Gaussian_Processes_Julia.ipynb)

Gaussian Processes have a lot of tunable hyperparameters, see `get_params(GP)`
for a list of parameters that can be [tuned](model_selection.md)

### DecisionTree.jl

- `DecisionTreeClassifier`
- `DecisionTreeRegressor`
- `RandomForestClassifier`
- `RandomForestRegressor`
- `AdaBoostStumpClassifier`

Documentation at [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl#scikitlearnjl) and in this [classification](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Classifier_Comparison_Julia.ipynb), examples in these [regression](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Decision_Tree_Regression_Julia.ipynb) notebooks.

### Contributing

To make your Julia model compatible with ScikitLearn.jl, you need to implement
the scikit-learn interface. See [ScikitLearnBase.jl](https://github.com/cstjean/ScikitLearnBase.jl)

