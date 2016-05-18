importall ScikitLearnBase


""" `TransformerFunction(f::Function)` is a non-learning transformer. It has a
no-op fit! function, and `transform(::TransformerFunction, X)` calls `f(X)`. It
is meant as a convenience for use in a pipeline. """
type TransformerFunction <: BaseEstimator
    f::Function
    TransformerFunction(; f=error("Missing argument f")) = new(f)
end
@declare_hyperparameters(TransformerFunction, [:f])
fit!(tf::TransformerFunction, X, y=nothing) = tf
transform(tf::TransformerFunction, X) = tf.f(X)
