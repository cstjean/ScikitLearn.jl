Pull requests are welcome. 

# TODO

Catching up with Python
-----

- Write tests (translate the relevant ones from scikit-learn)
- Support parallel execution. I'm not sure if njobs=2 is the right way to do this, but I don't have any experience with the Julia primitives
- Review docstrings. They still contain Python code.
- Support sparse matrices (grep for 'sparse' in /test). This might require work on PyCall.jl
- Model serialization (possibly by implementing JLD serialization for PyObjects
through pickling)

Supporting more models
------
There are three ways to make a library compatible with ScikitLearnBase.jl

- Make a PR to that library
- Create a new library that wraps the type and implements the interface
- Make a PR to ScikitLearn.jl, and use @require to avoid adding the library as a dependency

Option 1 is generally preferable where possible.

Interesting libraries
------
- Mocha.jl (or MXNet, Merlin). It's a good candidate for being a separate library (SkMocha.jl?). [sklearn-theano](https://github.com/sklearn-theano/sklearn-theano) is a good starting point.
- LowRankModels.jl
- RegERM.jl (looks maintained + has an SVM implementation)
- DiscriminantAnalysis.jl has an LDA and QDA implementation, and is currently
  maintained
- Clustering.jl
- GLM
- MixedModels (also consider FixedEffectModels)
- Mamba/Lora. Then we can have BayesianLinearRegression, BayesianHierarchicalModel, etc.
- XGBoost.jl - actively maintained
- BNP.jl is a recently-built package, pretty solid. Hard to tell if it will be
maintained, but looks great. Not released though.
- MultivariateStats.jl has some interesting algorithms. Eg. Lasso, ICA
- Make some NLP utilities available? TextAnalysis
- KernelDensity.jl
- OnlineAI.jl
- TopicModels has an LDA implementation. Not active, but still maintained
- BayesianDataFusion is a solid package, documented and recently maintained.
- DirichletProcessMixtures.jl would be cool, but not very active
- BayesNets.jl is interesting and currently in development
- Bootstrap.jl?? Is it possible to get bootstrap confidence intervals around arbitrary models?
- Curvefit.jl?

Hyperparameter tuning
-----
- Optim.jl. I don't think that could work, given that CV-performance is noisy.
- StochasticSearch.jl 
- Evolutionary.jl


Further plans
-----

- Make sklearn an optional dependency. We're not that far away...
- We should use Distributions.jl