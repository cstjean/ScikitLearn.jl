... work in progress

<img src="docs/example_images/Clustering_Comparison.png" alt="Some image" width="100">
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

You first need Python with numpy, which is easiest to get through
[Anaconda](https://www.continuum.io/downloads). Then [install
scikit-learn](http://scikit-learn.org/stable/install.html) with either

`conda install scikit-learn`

or 

`pip install -U scikit-learn`

(more instructions on [PyCall.jl](https://github.com/stevengj/PyCall.jl#installation)). Then you should `Pkg.clone` this package, as well as [ScikitLearnBase](https://github.com/cstjean/ScikitLearnBase.jl)

See the [notebooks folder](notebooks/) for examples.


