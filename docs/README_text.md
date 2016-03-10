# JukitLearn.jl

JukitLearn.jl implements the popular
[scikit-learn](http://scikit-learn.org/stable/) interface and algorithms in
Julia. It aims to provide the same functionality, while supporting both models
defined in Julia and those of the [scikit-learn
library](http://scikit-learn.org/stable/modules/classes.html) (via PyCall.jl).

Disclaimer: JukitLearn.jl borrows code and documentation from
[scikit-learn](http://scikit-learn.org/stable/), but it is not an official part
of that project. It is licensed under [BSD-3](LICENSE).

Main features:

- Around 150 machine learning and statistical models accessed through a uniform interface
- Pipelines and FeatureUnions
- Cross-validation
- Hyperparameter tuning

Check out the [Quick-Start
Guide](http://scikitlearnjl.readthedocs.org/en/latest/quickstart/) for a
tour.

## Installation

This package requires Python 2.7 with numpy, which is easiest to get through
[Anaconda](https://www.continuum.io/downloads), and [scikit-learn](http://scikit-learn.org/stable/install.html):

`conda install scikit-learn`

or 

`pip install -U scikit-learn`

(if you have issues, check out [PyCall.jl](https://github.com/stevengj/PyCall.jl#installation)). To install this package, at the Julia REPL type:

```julia
Pkg.clone("https://github.com/cstjean/JukitLearnBase.jl.git")
Pkg.clone("https://github.com/cstjean/JukitLearn.jl.git")
```

Finally, if you would like to run the examples, you will need [PyPlot.jl](https://github.com/stevengj/PyPlot.jl)

## Documentation

See the [manual](http://scikitlearnjl.readthedocs.org/en/latest/) and
[example gallery](docs/examples.md).

## Goal

JukitLearn.jl aims to achieve feature parity with scikit-learn. If you
encounter any problem that is solved by that library but not this one, [file an
issue](https://github.com/cstjean/JukitLearn.jl/issues).