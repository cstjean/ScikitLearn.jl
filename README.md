... work in progress

<a href="examples/Clustering_Comparison.ipynb"><img src="docs/example_images/Clustering_Comparison.png" alt="Some image" width="100"></a>
<img src="docs/example_images/Clustering_Comparison.png" alt="Another image" width="100">

# ScikitLearn.jl

ScikitLearn.jl allows you to train, combine, evaluate, and tune the parameters
of Julia machine-learning/statistical models, along with [scikit-learn
models](http://scikit-learn.org/stable/modules/classes.html) from the popular
Python library.

Disclaimer: ScikitLearn.jl is derived from
[scikit-learn](http://scikit-learn.org/stable/), but not an official part of
that project.

## Installation

This package requires Python 2.7 with numpy, which is easiest to get through
[Anaconda](https://www.continuum.io/downloads). Then [install
scikit-learn](http://scikit-learn.org/stable/install.html) with either

`conda install scikit-learn`

or 

`pip install -U scikit-learn`

(if you have issues, check out [PyCall.jl](https://github.com/stevengj/PyCall.jl#installation)). To install this package, at the Julia REPL type:

```julia
Pkg.clone("https://github.com/cstjean/ScikitLearn.jl.git")
Pkg.clone("https://github.com/cstjean/ScikitLearnBase.jl.git")
```

See the [examples folder](examples/) for more.
