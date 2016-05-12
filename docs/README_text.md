# ScikitLearn.jl

[![Documentation Status](https://readthedocs.org/projects/scikitlearnjl/badge/?version=latest)](http://scikitlearnjl.readthedocs.org/en/latest/?badge=latest)

ScikitLearn.jl implements the popular
[scikit-learn](http://scikit-learn.org/stable/) interface and algorithms in
Julia. It supports both models from the Julia ecosystem and those of the
[scikit-learn library](http://scikit-learn.org/stable/modules/classes.html)
(via PyCall.jl).

**Disclaimer**: ScikitLearn.jl borrows code and documentation from
[scikit-learn](http://scikit-learn.org/stable/), but *it is not an official part
of that project*. It is licensed under [BSD-3](LICENSE).

Main features:

- Around 150 [Julia](http://scikitlearnjl.readthedocs.org/en/latest/models/#Julia) and [Python](http://scikitlearnjl.readthedocs.io/en/latest/models/#python-models) models accessed through a uniform [interface](http://scikitlearnjl.readthedocs.org/en/latest/api/)
- [Pipelines and FeatureUnions](http://scikitlearnjl.readthedocs.org/en/latest/pipelines/)
- [Cross-validation](http://scikitlearnjl.readthedocs.org/en/latest/cross_validation/)
- [Hyperparameter tuning](http://scikitlearnjl.readthedocs.org/en/latest/model_selection/)
- [DataFrames support](http://scikitlearnjl.readthedocs.org/en/latest/dataframes/)

Check out the [Quick-Start
Guide](http://scikitlearnjl.readthedocs.org/en/latest/quickstart/) for a
tour.

## Installation

To install ScikitLearn.jl, run `Pkg.add("ScikitLearn")` at the REPL.

To import Python models (optional), ScikitLearn.jl requires [the scikit-learn Python library](http://scikitlearnjl.readthedocs.io/en/latest/models/#installation). Finally, some of the examples make use of [PyPlot.jl](https://github.com/stevengj/PyPlot.jl)

## Documentation

See the [manual](http://scikitlearnjl.readthedocs.org/en/latest/) and
[example gallery](docs/examples.md).

## Goal

ScikitLearn.jl aims to achieve feature parity with scikit-learn. If you
encounter any problem that is solved by that library but not this one, [file an
issue](https://github.com/cstjean/ScikitLearn.jl/issues).