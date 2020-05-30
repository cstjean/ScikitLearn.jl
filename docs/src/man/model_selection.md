# Model Selection

## Grid Search

Most models contain _hyperparameters_: parameters that are specified in the
constructor, and not learned from the data. ScikitLearn.jl provides
`GridSearchCV` to find the best set of hyper-parameter:

```jldoctest
julia> using ScikitLearn, Random

julia> Random.seed!(11);

julia> X = rand(25,4); y = rand([true,false], 25);

julia> @sk_import linear_model: LogisticRegression;

julia> using ScikitLearn.GridSearch: GridSearchCV

julia> gridsearch = GridSearchCV(LogisticRegression(max_iter=200), Dict(:C => 0.1:0.1:2.0))
GridSearchCV
  estimator: PyCall.PyObject
  param_grid: Dict{Symbol,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
  scoring: Nothing nothing
  loss_func: Nothing nothing
  score_func: Nothing nothing
  fit_params: Dict{Any,Any}
  n_jobs: Int64 1
  iid: Bool true
  refit: Bool true
  cv: Nothing nothing
  verbose: Int64 0
  error_score: String "raise"
  scorer_: Nothing nothing
  best_params_: Nothing nothing
  best_score_: Nothing nothing
  grid_scores_: Nothing nothing
  best_estimator_: Nothing nothing



julia> fit!(gridsearch, X, y)
GridSearchCV
  estimator: PyCall.PyObject
  param_grid: Dict{Symbol,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
  scoring: Nothing nothing
  loss_func: Nothing nothing
  score_func: Nothing nothing
  fit_params: Dict{Any,Any}
  n_jobs: Int64 1
  iid: Bool true
  refit: Bool true
  cv: Nothing nothing
  verbose: Int64 0
  error_score: String "raise"
  scorer_: score (function of type typeof(score))
  best_params_: Dict{Symbol,Any}
  best_score_: Float64 0.6
  grid_scores_: Array{ScikitLearn.Skcore.CVScoreTuple}((20,))
  best_estimator_: PyCall.PyObject



julia> println("Best hyper-parameters: $(gridsearch.best_params_)")
Best hyper-parameters: Dict{Symbol,Any}(:C => 0.8)

```

See `?GridSearchCV` and the [scikit-learn docs](http://scikit-learn.org/stable/modules/grid_search.html) for details.

### Examples

- [Quick start guide](quickstart.md)
- [Pipelining: chaining a PCA and a logistic regression](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Pipeline_PCA_Logistic.ipynb)
- [Concatenating multiple feature extraction methods](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Feature_Stacker.ipynb)

## Random Search

`RandomizedSearchCV` will sample from each parameter independently.
Documentation [here](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html).

Note: The distributions have to be specified using `scipy.stats` (see example
below), but we hope to support Distributions.jl in the future. File an issue if
this is a pain point.

### Examples

- [Comparing randomized search and grid search for hyperparameter estimation](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Randomized_Search.ipynb)

