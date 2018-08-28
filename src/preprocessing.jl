using IterTools: chain
import IterTools
import Compat


""" `TransformerFunction(f::Function)` is a non-learning transformer. It has a
no-op fit! function, and `transform(::TransformerFunction, X)` calls `f(X)`. It
is meant as a convenience for use in a pipeline. """
mutable struct TransformerFunction <: BaseEstimator
    f::Function
    TransformerFunction(; f=error("Missing argument f")) = new(f)
end
@declare_hyperparameters(TransformerFunction, [:f])
fit!(tf::TransformerFunction, X, y=nothing) = tf
transform(tf::TransformerFunction, X) = tf.f(X)

""" `TypeConverter(; new_type=Float64)` will transform the input array from
one type to another (useful in pipelines) """
TypeConverter(; new_type::Type{T}=Float64) where T =
    TransformerFunction(; f=arr->convert(Array{T}, arr))

################################################################################

# Test TODO: assert that these lines yield the same results for scikit-learn
## df = dataset("datasets", "iris")
## DataFrames.head(df)
## p1 = ScikitLearn.Preprocessing.PolynomialFeatures(degree=2,include_bias=false,interaction_only=false)
## mean(fit_transform!(p1, convert(Array, df[1:10, 1:4])))


"""    PolynomialFeatures(; degree=2, interaction_only=false, include_bias=true)

Generate polynomial and interaction features.
Generate a new feature matrix consisting of all polynomial combinations
of the features with degree less than or equal to the specified degree.
For example, if an input sample is two dimensional and of the form
[a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

Parameters
----------

- `degree::Int`
    The degree of the polynomial features. Default = 2.
- `interaction_only::Bool`, default = False
    If true, only interaction features are produced: features that are
    products of at most `degree` *distinct* input features (so not
    `x[1] ^ 2`, `x[0] * x[2] ^ 3`, etc.).
- `include_bias::Bool`
    If True (default), then include a bias column, the feature in which
    all polynomial powers are zero (i.e. a column of ones - acts as an
    intercept term in a linear model).

Examples
--------

```julia
# Python code - TODO: translate
>>> X = np.arange(6).reshape(3, 2)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> poly = PolynomialFeatures(2)
>>> poly.fit_transform(X)
array([[  1.,   0.,   1.,   0.,   0.,   1.],
       [  1.,   2.,   3.,   4.,   6.,   9.],
       [  1.,   4.,   5.,  16.,  20.,  25.]])
>>> poly = PolynomialFeatures(interaction_only=True)
>>> poly.fit_transform(X)
array([[  1.,   0.,   1.,   0.],
       [  1.,   2.,   3.,   6.],
       [  1.,   4.,   5.,  20.]])
```

Attributes
----------

- `powers_::Array`, shape (n_input_features, n_output_features)
    powers_[i, j] is the exponent of the jth input in the ith output.
- `n_input_features_::Int`
    The total number of input features.
- `n_output_features_::Int`
    The total number of polynomial output features. The number of output
    features is computed by iterating over all suitably sized combinations
    of input features.

Notes
-----

Be aware that the number of features in the output array scales
polynomially in the number of features of the input array, and
exponentially in the degree. High degrees can cause overfitting.
See :ref:`examples/linear_model/plot_polynomial_interpolation.py
<example_linear_model_plot_polynomial_interpolation.py>`
"""
mutable struct PolynomialFeatures <: BaseEstimator
    degree::Int
    interaction_only::Bool
    include_bias::Bool
    n_input_features_::Int
    n_output_features_::Int
    PolynomialFeatures(; degree=2, interaction_only=false, include_bias=true) =
        new(degree, interaction_only, include_bias)
end
@declare_hyperparameters(PolynomialFeatures,
                         [:degree, :interaction_only, :include_bias])

combinations_with_replacement(arr, degree) =
    (degree == 0 ? [()] : # needs to special-case
     filter(issorted, IterTools.product(fill(arr, degree)...)))

function _combinations(n_features, degree, interaction_only, include_bias)
    comb = interaction_only ? combinations : combinations_with_replacement
    start = Int(!include_bias)
    return chain([comb(0:n_features-1, i) for i in start:degree]...)
end

function powers_(self::PolynomialFeatures)
    never_tested() # TODO
    combinations = _combinations(self.n_input_features_, self.degree,
                                 self.interaction_only,
                                 self.include_bias)
    return vcat([bincount(c+1, minlength=self.n_input_features_)
                 for c in combinations]...)
end

function fit!(self::PolynomialFeatures, X, y=nothing)
    # Compute number of output features.
    n_samples, n_features = size(X)
    combinations = _combinations(n_features, self.degree,
                                 self.interaction_only,
                                 self.include_bias)
    self.n_input_features_ = n_features
    self.n_output_features_ = count(_->true, combinations)
    return self
end

function transform(self::PolynomialFeatures,
                   X::AbstractArray{T}, y=nothing) where {T<:AbstractFloat}
    ## Transform data to polynomial features
    ## Parameters
    ## ----------
    ## X : array-like, shape [n_samples, n_features]
    ##     The data to transform, row by row.
    ## Returns
    ## -------
    ## XP : np.ndarray shape [n_samples, NP]
    ##     The matrix of features, where NP is the number of polynomial
    ##     features generated from the combination of inputs.
    n_samples, n_features = size(X)

    if n_features != self.n_input_features_
        throw(DimensionMismatch("X shape does not match training shape"))
    end

    # allocate output data
    XP = Matrix{T}(undef, n_samples, self.n_output_features_)
    
    combinations = _combinations(n_features, self.degree,
                                 self.interaction_only,
                                 self.include_bias)
    for (i, c) in enumerate(combinations)
        XP[:, i] = prod(X[:, Int[k+1 for k in c]], 2)
    end
    return XP
end
