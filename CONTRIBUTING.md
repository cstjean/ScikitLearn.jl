Pull requests are welcome. 

TODO
----

- Write tests (translate the relevant ones from scikit-learn)
- Support parallel execution. I'm not sure if njobs=2 is the right way to do this, but I don't any experience with the Julia primitives
- Dataframes. [sklearn-pandas](https://github.com/paulgb/sklearn-pandas) looks like a good model (and it should probably be in a separate library, to avoid the dependency)
- Write more example notebooks
- Review docstrings. They still contain Python code.
- Support sparse matrices (grep for 'sparse' in /test). This might require work on PyCall.jl
- Model serialization (possibly by implementing JLD serialization for PyObjects
through pickling)

I'd like to support Mocha (or some other deep learning library) at some point.
It's a good candidate for being a separate library (SkMocha.jl?). [sklearn-theano](https://github.com/sklearn-theano/sklearn-theano) is a good starting point.

Supporting more models
------
There are three ways to make a library compatible with ScikitLearnBase.jl

- Make a PR to that library (file an issue first to see if there's an interest)
- Create a new library that wraps the type and implements the interface
- Make a PR to ScikitLearn.jl, and use @require to avoid adding the library as a dependency

Option 1 is generally preferable where possible.