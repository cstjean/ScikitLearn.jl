Saving models to disk
------

[JLD.jl](https://github.com/JuliaIO/JLD.jl) is the preferred way of saving
`ScikitLearn.jl` models. If you also use Python models (via
`@sk_import`), you will have to import
[PyCallJLD](https://github.com/JuliaPy/PyCallJLD.jl) as well.

```julia
using PyCall, JLD, PyCallJLD
using ScikitLearn
using ScikitLearn.Pipelines
@sk_import decomposition: PCA
@sk_import linear_model: LinearRegression

pca = PCA()
lm = LinearRegression()

X=rand(10, 3); y=rand(10);

pip = Pipeline([("PCA", pca), ("LinearRegression", lm)])
fit!(pip, X, y)   # fit to some dataset

JLD.save("pipeline.jld", "pip", pip)
pip = JLD.load("pipeline.jld", "pip") # Load back the pipeline

```
