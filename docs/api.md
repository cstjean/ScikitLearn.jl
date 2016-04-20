Api
------

Not all models implement every function. All input matrices X are vertical
(of shape `(n_sample, n_feature)`).

#### fit!

`fit!(model, X)` ; `fit!(model, X, y)`

Trains `model` on the input data `X` and `y` (for supervised learning) or on
just `X` (for unsupervised learning).  The `model` object is always returned,
allowing code like `classifier = fit!(LogisticRegression(), X, y)`

#### partial_fit!

`partial_fit!(model, X)` ; `partial_fit!(model, X, y)`

Incrementally trains model on the new data `X` and `y`. For instance, this
might perform a stochastic gradient descent step.

#### predict

`predict(model, X)` returns the predicted class of each row in `X` (for
classifiers) or the predicted value (for regressors).

#### predict_proba

`predict_proba(model, X)` returns an `(N, C)` matrix containing the probability
that the n_th sample belongs to the c_th class.

#### predict_log_proba

`predict_log_proba(model, X)` is equivalent to `log(predict_proba(model, X))`
but can be either more accurate (for small probabilities) or faster (avoiding the
exponential).

#### transform

For unsupervised learning models and for preprocessing, `transform(model, X)`
applies the transformation from `model` to `X`, and returns a similar array
(same number of rows, possibly different number of columns).

#### fit_transform!

`fit_transform!(model, X)` is equivalent to `transform(fit!(model, X), X)` but
can sometimes be more efficient.

#### inverse_transform

`inverse_transform(model, X)` applies the inverse of the model transformation.

#### score_samples

`score_samples(model, X)` evaluates the density model on X.

#### score

`score(model, X)` and `score(model, X, y)` assign a score to how likely `X` or
`y|X` is given the learned model parameters. The higher this score is, the better
the model. This is used for cross-validation.

#### decision_function

`decision_function(model, X)` returns the distance of the samples to the
decision boundary.

## Model Internals

- `clone(model)` returns a new object of the same type as model, with the same
  hyperparameters, but unfit.
- `set_params!(model, param1=value1, param2=value2, ...)` changes the model
  hyperparameters.
- `get_params(model)` returns all the model hyperparameters that can be
  changed with `set_params!`
- `is_classifier(model)` is true if `model` is a classifier.
- `get_feature_names(model)` returns the name of the output features
- `is_pairwise(model)`: WRITEME
- `get_classes(preprocessor)` returns the label of each class