Most data science and machine learning problems involve several steps of
data preprocessing and transformation. ScikitLearn.jl provides two types to
facilitate this task.

Pipelines
----

Pipeline can be used to chain multiple estimators into one. This is useful as
there is often a fixed sequence of steps in processing the data, for example
feature selection, normalization and classification.

```julia
using ScikitLearn
using ScikitLearn.Pipelines: Pipeline, make_pipeline
@sk_import decomposition: PCA

estimators = [("reduce_dim", PCA()), ("logistic_regression", LogisticRegression())]
clf = Pipeline(estimators)
fit!(clf, X, y)
```

See `?Pipeline`, `?make_pipeline` and the [user guide](http://scikit-learn.org/stable/modules/pipeline.html) for details.

#### Examples

- [Pipelining: chaining a PCA and a logistic regression](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Pipeline_PCA_Logistic.ipynb)
- [Restricted Boltzmann Machine features for digit classification](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/RBM.ipynb)

Feature Unions
----

FeatureUnion combines several transformer objects into a new transformer that
combines their output. A FeatureUnion takes a list of transformer
objects. During fitting, each of these is fit to the data independently. For
transforming data, the transformers are applied in parallel, and the sample
vectors they output are concatenated end-to-end into larger vectors.

```julia
using ScikitLearn.Pipelines: FeatureUnion
@sk_import decomposition: (PCA, KernelPCA)

estimators = [("linear_pca", PCA()), ("kernel_pca", KernelPCA())]
combined = FeatureUnion(estimators)
```

See `?FeatureUnion` and [the user guide](http://scikit-learn.org/stable/modules/pipeline.html#featureunion-composite-feature-spaces) for more.

#### Examples

- [Concatenating multiple feature extraction methods](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Feature_Stacker.ipynb)
