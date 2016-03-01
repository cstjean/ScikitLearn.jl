# ScikitLearn.jl

ScikitLearn.jl allows you to train, combine, evaluate, and tune the parameters
of Julia machine-learning/statistical models along with the scikit-learn
[models](http://scikit-learn.org/stable/modules/classes.html) from the popular
Python library.

ScikitLearn.jl is derived from [scikit-learn](http://scikit-learn.org/stable/),
but not an official part of that project.

## Installation

You need the Python version of scikit-learn first, which you can get from either

`conda install scikit-learn`

or 

`pip install -U scikit-learn`

(more instructions [here](http://scikit-learn.org/stable/install.html) and on [PyCall.jl](https://github.com/stevengj/PyCall.jl#installation)). Then
you should `Pkg.clone` this package, as well as [SklearnBase]

See the `notebooks` folder for examples in Julia.


```
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
```
