Saving models to disk
------

[JLD.jl](https://github.com/JuliaIO/JLD.jl) is the preferred way of saving Julia objects,
including trained `ScikitLearn.jl` models. If you also use Python models (via
`@sk_import`), you will have to import
[PyCallJLD](https://github.com/JuliaPy/PyCallJLD.jl) as well.

```julia
using ScikitLearn
using ScikitLearn.Pipelines
using PyCall, JLD, PyCallJLD

@sk_import decomposition: PCA
@sk_import linear_model: LinearRegression

pca = PCA()
lm = LinearRegression()

pip = Pipeline([("PCA", pca), ("LinearRegression", lm)])

JLD.save("pipeline.jld", "pip", pip)

# Load back the pipeline
pip = JLD.load("pipeline.jld", "pip")
```