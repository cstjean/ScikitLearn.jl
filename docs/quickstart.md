Quick start guide
-----

This tutorial doesn't assume any familiarity with scikit-learn, but you do
need to know about machine learning.

First, let's load the classic iris dataset. If you don't have RDatasets,
`Pkg.add` it.

```julia
using RDatasets: dataset

iris = dataset("datasets", "iris")

X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])
```

We convert the data to array form, because ScikitLearn.jl does not (yet) support
dataframes.

Next, we will load LogisticRegression from scikit-learn's [library](models.md).

```julia
using ScikitLearn

@sk_import linear_model: LogisticRegression
```

Every model's constructor accepts hyper-parameters (such as regression
strength, whether to fit the intercept, the penalty type, etc.) as
keyword-arguments.  Check out `?LogisticRegression` for details.

```julia
model = LogisticRegression(fit_intercept=true)
```

Then we train the model and evaluate its accuracy on the training set:

```julia
fit!(model, X, y)

accuracy = sum(predict(model, X) .== y) / length(y)
println("accuracy: $accuracy")

> accuracy: 0.96
```

### Cross-validation

This will train five models, on five train/test splits of X and y, and return
the accuracy of each:

```julia
using ScikitLearn.CrossValidation: cross_val_score

cross_val_score(LogisticRegression(), X, y; cv=5)  # 5-fold

> 5-element Array{Float64,1}:
>  1.0     
>  0.966667
>  0.933333
>  0.9     
>  1.0     
```

See this [tutorial](http://scikit-learn.org/stable/modules/cross_validation.html) for more information.

### Hyper-parameter tuning

`LogisticRegression` has a regularization-strength parameter `C` (smaller is
stronger). We can use grid search algorithms to find the optimal `C`.

`GridSearchCV` will try all values of `C` in `0.1:0.1:2.0` and will
return the one with the highest cross-validation performance.

```julia
using ScikitLearn.GridSearch: GridSearchCV

gridsearch = GridSearchCV(LogisticRegression(), Dict(:C => 0.1:0.1:2.0))
fit!(gridsearch, X, y)
println("Best parameters: $(gridsearch.best_params_)")

> Best parameters: Dict{Symbol,Any}(:C=>1.1)
```

We can plot cross-validation accuracy vs. `C`

```julia
using PyPlot

plot([cv_res.parameters[:C] for cv_res in gridsearch.grid_scores_],
     [mean(cv_res.cv_validation_scores) for cv_res in gridsearch.grid_scores_])
```

Note that it is good statistical practice to keep a separate test set.