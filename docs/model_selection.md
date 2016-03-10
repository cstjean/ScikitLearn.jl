Model Selection
------

Most models contain _hyperparameters_: parameters that are specified in the
constructor, and not learned from the data. ScikitLearn.jl provides
`GridSearchCV` to find the best set of hyper-parameter:

```julia
using ScikitLearn.GridSearch: GridSearchCV

gridsearch = GridSearchCV(LogisticRegression(), Dict(:C => 0.1:0.1:2.0))
fit!(gridsearch, X, y)
println("Best hyper-parameters: $(gridsearch.best_params_)")
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