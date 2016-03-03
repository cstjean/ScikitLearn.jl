# ScikitLearn.jl

ScikitLearn.jl is a Julia machine learning library modeled after
[scikit-learn](http://scikit-learn.org/stable/). It supports both models defined
in Julia and those of the [scikit-learn library](http://scikit-learn.org/stable/modules/classes.html).

Disclaimer: ScikitLearn.jl is derived and borrows code from
[scikit-learn](http://scikit-learn.org/stable/), but it is not an official part
of that project. It is licensed under [BSD-3](LICENSE).

## Installation

This package requires Python 2.7 with numpy, which is easiest to get through
[Anaconda](https://www.continuum.io/downloads), and [scikit-learn](http://scikit-learn.org/stable/install.html):

`conda install scikit-learn`

or 

`pip install -U scikit-learn`

(if you have issues, check out [PyCall.jl](https://github.com/stevengj/PyCall.jl#installation)). To install this package, at the Julia REPL type:

```julia
Pkg.clone("https://github.com/cstjean/ScikitLearn.jl.git")
Pkg.clone("https://github.com/cstjean/ScikitLearnBase.jl.git")
```

## Examples

See the notebooks in the [example folder](examples/).
