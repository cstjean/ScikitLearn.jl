# Models

## Python models

scikit-learn has on the order of 100 to 200 models (more generally called
"estimators"), split into three categories:

- [Supervised Learning](http://scikit-learn.org/stable/supervised_learning.html) (linear regression, support vector machines, random forest, neural nets, ...)
- [Unsupervised Learning](http://scikit-learn.org/stable/unsupervised_learning.html) (clustering, PCA, mixture models, manifold learning, ...)
- [Dataset Transformation](http://scikit-learn.org/stable/data_transforms.html) (preprocessing, text feature extraction, one-hot encoding, ...)

All of those estimators will work with ScikitLearn.jl. They are imported with
`@sk_import`. For example, here's how to import and fit
`sklearn.linear_regression.LogisticRegression`

```jldoctest models
julia> using ScikitLearn, Random

julia> Random.seed!(11); #ensures reproducibility

julia> X = rand(20,3); y = rand([true, false], 20);

julia> @sk_import linear_model: LogisticRegression
PyObject <class 'sklearn.linear_model._logistic.LogisticRegression'>

julia> using ScikitLearn.CrossValidation: train_test_split

julia> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42);

julia> log_reg = fit!(LogisticRegression(penalty="l2"), X_train, y_train);

julia> predict(log_reg, X_test)
5-element Array{Bool,1}:
 0
 0
 0
 0
 0

```

Reminder: `?LogisticRegression` contains a lot of information about the model
parameters.

### Installation and importing Python models

Importing the Python models requires Python 3.x with numpy, and the
scikit-learn library. This is easiest to get through [Conda.jl](https://github.com/Luthaf/Conda.jl), which is already
installed on your system.  Calling `@sk_import linear_model: LinearRegression` should automatically install everything. You can also install `scikit-learn`
manually with `Conda.add("scikit-learn")`. If you have other issues, please
refer to [PyCall.jl](https://github.com/stevengj/PyCall.jl#installation), or
[post an issue](https://github.com/cstjean/ScikitLearn.jl/issues/new)



## Julia models

Julia models are hosted in other packages, and need to be installed separately
with `Pkg.add` or `Pkg.checkout` (to get the latest version - sometimes
necessary). They all implement the [common api](api.md), and provide
hyperparameter information in their `?docstrings`.

!!! note

    Unfortunately, some packages export a `fit!` function that conflicts with
    ScikitLearn's `fit!`. This can be fixed by adding this line:
    
    ```julia
    using ScikitLearn: fit!, predict
    
    ```
    
### ScikitLearn in-built models

- `ScikitLearn.Models.LinearRegression()` implements linear regression using
  `\`, optimized for speed. See `?LinearRegression` for fitting options.

### GaussianMixtures.jl

```jldoctest models
julia> using GaussianMixtures: GMM #remember to install package first

julia> gmm = fit!(GMM(n_components=3, kind=:diag), X_train);
[ Info: Initializing GMM, 3 Gaussians diag covariance 3 dimensions using 15 data points
  Iters               objv        objv-change | affected
-------------------------------------------------------------
      0       1.462249e+00
      1       1.041033e+00      -4.212161e-01 |        2
      2       9.589243e-01      -8.210827e-02 |        2
      3       9.397430e-01      -1.918135e-02 |        0
      4       9.397430e-01       0.000000e+00 |        0
K-means converged with 4 iterations (objv = 0.9397430000827904)
┌ Info: K-means with 15 data points using 4 iterations
└ 1.3 data points per parameter

julia> predict_proba(gmm, X_test)
5×3 Array{Float64,2}:
 1.37946e-7   5.58899e-9   1.0
 0.986895     1.98749e-10  0.0131053
 0.998037     1.00296e-15  0.00196321
 2.66238e-11  0.041746     0.958254
 0.999984     4.05443e-6   1.16204e-5

```

Documentation at [GaussianMixtures.jl](https://github.com/davidavdav/GaussianMixtures.jl). Example: [density estimation](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Density_Estimation_Julia.ipynb)

### GaussianProcesses.jl

```jldoctest models
julia> using GaussianProcesses: GPE, MeanZero, SE #remember to install package first

julia> gp = fit!(GPE(; mean=MeanZero(), kernel=SE(0.0, 0.0), logNoise=-1e8), X_train, Float64.(y_train))
GP Exact object:
  Dim = 3
  Number of observations = 15
  Mean function:
    Type: MeanZero, Params: Float64[]
  Kernel:
    Type: GaussianProcesses.SEIso{Float64}, Params: [0.0, 0.0]
  Input observations =
[0.376913304113047 0.5630896022795546 … 0.31598998347835017 0.5828199336036355; 0.50060556533132 0.4124482236437548 … 0.6750380496244157 0.6147514739028759; 0.5142063690337368 0.4774433498612982 … 0.9823652195180261 0.21010382988916376]
  Output observations = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
  Variance of observation noise = 0.0
  Marginal Log-Likelihood = -749.473

julia> predict(gp, X_test)
5-element Array{Float64,1}:
  2.1522493172851114
  1.298965158590363
  0.8142639915887457
 -0.7287701449370729
  0.7495235968268048

```

Documentation at [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl) and in the `?GPE` docstring. Example: [Gaussian Processes](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Gaussian_Processes_Julia.ipynb)

Gaussian Processes have a lot of hyperparameters, see `get_params(GP)`
for a list. They can all be [tuned](model_selection.md)

### DecisionTree.jl

- `DecisionTreeClassifier`
- `DecisionTreeRegressor`
- `RandomForestClassifier`
- `RandomForestRegressor`
- `AdaBoostStumpClassifier`

Documentation at [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl#scikitlearnjl). Examples: [Classifier Comparison](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Classifier_Comparison_Julia.ipynb), [Decision Tree Regression](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Decision_Tree_Regression_Julia.ipynb) notebooks.

### LowRankModels.jl

- `SkGLRM`: Generalized Low Rank Model
- `PCA`: Principal Component Analysis
- `QPCA`: Quadratically Regularized PCA
- `RPCA`: Robust PCA
- `NNMF`: Non-negative matrix factorization
- `KMeans`: The k-means algorithm

!!! note
    These algorithms are all special cases of the Generalized Low Rank Model algorithm,
    whose main goal is to provide flexible loss and regularization for heterogeneous data. 
    Specialized algorithms will achieve faster convergence in general.
    Documentation at [LowRankModels.jl](https://github.com/madeleineudell/LowRankModels.jl#scikitlearn). Example: [KMeans Digit Classifier](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Plot_Kmeans_Digits_Julia.ipynb).



## Contributing new models

To make your Julia model compatible with ScikitLearn.jl, you need to implement
the scikit-learn interface. See [ScikitLearnBase.jl](https://github.com/cstjean/ScikitLearnBase.jl)

