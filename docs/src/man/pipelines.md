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
 ("reduce_dim", PyObject PCA())
 ("logistic_regression", PyObject LogisticRegression())
 
julia> clf = Pipeline(estimators)
Pipeline(Tuple{Any,Any}[("reduce_dim", PyObject PCA()), ("logistic_regression", PyObject LogisticRegression())], Any[PyObject PCA(), PyObject LogisticRegression()])

julia> fit!(clf, X, y)
 Pipeline(Tuple{Any,Any}[("reduce_dim", PyObject PCA()), ("logistic_regression", PyObject LogisticRegression())], Any[PyObject PCA(), PyObject LogisticRegression()])
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
 ("linear_pca", PyObject PCA())
 ("kernel_pca", PyObject KernelPCA())

julia> combined = FeatureUnion(estimators)
FeatureUnion(Tuple{Any,Any}[("linear_pca", PyObject PCA()), ("kernel_pca", PyObject KernelPCA())], 1, nothing)

```

See `?FeatureUnion` and [the user guide](http://scikit-learn.org/stable/modules/pipeline.html#featureunion-composite-feature-spaces) for more.

### Examples

- [Concatenating multiple feature extraction methods](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Feature_Stacker.ipynb)
