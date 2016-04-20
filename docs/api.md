Api
------

Not all models implement every function. All input matrices X are vertical
(of shape `(n_example, n_feature)`).

#### fit!

`fit!(model, X)` ; `fit!(model, X, y)`

Trains `model` on the input data `X` and `y` (for supervised learning) or on
just `X` (for unsupervised learning).  The `model` object is always returned,
allowing code like `classifier = fit!(LogisticRegression(), X, y)`

#### partial_fit

`partial_fit!(model, X)` ; `partial_fit!(model, X, y)`

Incrementally trains model on the new data `X` and `y`. For instance, this
might perform a stochastic gradient descent step.

transform, fit_transform!,
             predict, predict_proba, predict_log_proba,
             score_samples, sample,
             score, decision_function, clone, set_params!,
             get_params, is_classifier, is_pairwise,
             get_feature_names, get_classes,
             inverse_transform)
