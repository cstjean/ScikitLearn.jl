 <a href="./examples/Clustering_Comparison.ipynb"><img src="./docs/example_images/Clustering_Comparison.png" alt="# Comparing different clustering algorithms on toy datasets" width="170"> </a>  <a href="./examples/Density_Estimation_Julia.ipynb"><img src="./docs/example_images/Density_Estimation_Julia.png" alt="# Density Estimation for a mixture of Gaussians (using GaussianMixtures.jl)" width="170"> </a>  <a href="./examples/Feature_Stacker.ipynb"><img src="./docs/example_images/Text_image.png" alt="# Concatenating multiple feature extraction methods" width="170"> </a>  <a href="./examples/Outlier_Detection.ipynb"><img src="./docs/example_images/Outlier_Detection.png" alt="# Outlier detection with several methods" width="170"> </a>  <a href="./examples/Pipeline_PCA_Logistic.ipynb"><img src="./docs/example_images/Pipeline_PCA_Logistic.png" alt="# Pipelining: chaining a PCA and a logistic regression" width="170"> </a>  <a href="./examples/RBM.ipynb"><img src="./docs/example_images/RBM.png" alt="# Restricted Boltzmann Machine features for digit classification" width="170"> </a>  <a href="./examples/Simple_1D_Kernel_Density.ipynb"><img src="./docs/example_images/Simple_1D_Kernel_Density.png" alt="# Simple 1D Kernel Density Estimation" width="170"> </a>  <a href="./examples/Text_Feature_Extraction.ipynb"><img src="./docs/example_images/Text_image.png" alt="# Sample pipeline for text feature extraction and evaluation" width="170"> </a>  <a href="./examples/Two_Class_Adaboost.ipynb"><img src="./docs/example_images/Two_Class_Adaboost.png" alt="# Two Class Adaboost" width="170"> </a>  <a href="./examples/Underfitting_vs_Overfitting.ipynb"><img src="./docs/example_images/Underfitting_vs_Overfitting.png" alt="# Underfitting vs. Overfitting" width="170"> </a> 

# ScikitLearn.jl

[![Documentation Status](https://readthedocs.org/projects/scikitlearnjl/badge/?version=latest)](http://scikitlearnjl.readthedocs.org/en/latest/?badge=latest)

ScikitLearn.jl implements the popular
[scikit-learn](http://scikit-learn.org/stable/) interface and algorithms in
Julia. It aims to provide the same functionality, while supporting both models
defined in Julia and those of the [scikit-learn
library](http://scikit-learn.org/stable/modules/classes.html) (via PyCall.jl).

**Disclaimer**: ScikitLearn.jl borrows code and documentation from
[scikit-learn](http://scikit-learn.org/stable/), but *it is not an official part
of that project*. It is licensed under [BSD-3](LICENSE).

Main features:

- Around 150 machine learning and statistical [models](http://scikitlearnjl.readthedocs.org/en/latest/models/) accessed through a uniform [interface](http://scikitlearnjl.readthedocs.org/en/latest/api/)
- [Pipelines and FeatureUnions](http://scikitlearnjl.readthedocs.org/en/latest/pipelines/)
- [Cross-validation](http://scikitlearnjl.readthedocs.org/en/latest/cross_validation/)
- [Hyperparameter tuning](http://scikitlearnjl.readthedocs.org/en/latest/model_selection/)
- [DataFrames support](http://scikitlearnjl.readthedocs.org/en/latest/dataframes/)

Check out the [Quick-Start
Guide](http://scikitlearnjl.readthedocs.org/en/latest/quickstart/) for a
tour.

## Installation

This package requires Python 2.7 with numpy, which is easiest to get through
[Anaconda](https://www.continuum.io/downloads), and [scikit-learn](http://scikit-learn.org/stable/install.html):

`conda install scikit-learn`

or 

`pip install -U scikit-learn`

(if you have issues, check out [PyCall.jl](https://github.com/stevengj/PyCall.jl#installation)). To install ScikitLearn.jl, 

```julia
Pkg.add("ScikitLearn")
```

Finally, if you would like to run the examples, you will need [PyPlot.jl](https://github.com/stevengj/PyPlot.jl)

## Documentation

See the [manual](http://scikitlearnjl.readthedocs.org/en/latest/) and
[example gallery](docs/examples.md).

## Goal

ScikitLearn.jl aims to achieve feature parity with scikit-learn. If you
encounter any problem that is solved by that library but not this one, [file an
issue](https://github.com/cstjean/ScikitLearn.jl/issues).