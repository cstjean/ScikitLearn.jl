Most data science and machine learning problems involve several steps of
data preprocessing and transformation. ScikitLearn.jl provides two types to
facilitate this task.

## Pipelines


Pipeline can be used to chain multiple estimators into one. This is useful as
there is often a fixed sequence of steps in processing the data, for example
feature selection, normalization and classification.

```jldoctest pipelines
julia> using ScikitLearn

julia> using ScikitLearn.Pipelines: Pipeline, make_pipeline

julia> @sk_import decomposition: PCA
PyObject <class 'sklearn.decomposition._pca.PCA'>

julia> @sk_import linear_model: LogisticRegression 
PyObject <class 'sklearn.linear_model._logistic.LogisticRegression'>

julia> using RDatasets: dataset

julia> iris = dataset("datasets", "iris");

julia> X = convert(Array, iris[!, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]]);

julia> y = convert(Array, iris[!,:Species]);

julia> estimators = [("reduce_dim", PCA()), ("logistic_regression", LogisticRegression())]
2-element Array{Tuple{String,PyCall.PyObject},1}:
 ("reduce_dim", PyObject PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False))
 ("logistic_regression", PyObject LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False))

julia> clf = Pipeline(estimators)
Pipeline(Tuple{Any,Any}[("reduce_dim", PyObject PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)), ("logistic_regression", PyObject LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False))], Any[PyObject PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False), PyObject LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)])

julia> fit!(clf, X, y)
Pipeline(Tuple{Any,Any}[("reduce_dim", PyObject PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)), ("logistic_regression", PyObject LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False))], Any[PyObject PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False), PyObject LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)])
```

See `?Pipeline`, `?make_pipeline` and the [user guide](http://scikit-learn.org/stable/modules/pipeline.html) for details.

### Examples

- [Pipelining: chaining a PCA and a logistic regression](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Pipeline_PCA_Logistic.ipynb)
- [Restricted Boltzmann Machine features for digit classification](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/RBM.ipynb)

## Feature Unions

FeatureUnion combines several transformer objects into a new transformer that
combines their output. A FeatureUnion takes a list of transformer
objects. During fitting, each of these is fit to the data independently. For
transforming data, the transformers are applied in parallel, and the sample
vectors they output are concatenated end-to-end into larger vectors.

```jldoctest pipelines
julia> using ScikitLearn.Pipelines: FeatureUnion

julia> @sk_import decomposition: KernelPCA
PyObject <class 'sklearn.decomposition._kernel_pca.KernelPCA'>

julia> estimators = [("linear_pca", PCA()), ("kernel_pca", KernelPCA())]
2-element Array{Tuple{String,PyCall.PyObject},1}:
 ("linear_pca", PyObject PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False))
 ("kernel_pca", PyObject KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
          fit_inverse_transform=False, gamma=None, kernel='linear',
          kernel_params=None, max_iter=None, n_components=None, n_jobs=None,
          random_state=None, remove_zero_eig=False, tol=0))

julia> combined = FeatureUnion(estimators)
FeatureUnion(Tuple{Any,Any}[("linear_pca", PyObject PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)), ("kernel_pca", PyObject KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
          fit_inverse_transform=False, gamma=None, kernel='linear',
          kernel_params=None, max_iter=None, n_components=None, n_jobs=None,
          random_state=None, remove_zero_eig=False, tol=0))], 1, nothing)

```

See `?FeatureUnion` and [the user guide](http://scikit-learn.org/stable/modules/pipeline.html#featureunion-composite-feature-spaces) for more.

### Examples

- [Concatenating multiple feature extraction methods](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Feature_Stacker.ipynb)
