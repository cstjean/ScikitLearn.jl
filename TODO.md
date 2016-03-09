Re. Module loading: I don't have a great solution yet. I'm leaning towards
having a separate module for each main Python module
(eg. `sk_linear_model`). Or maybe I can @pyimport everything to make the
hierarchy work and use `@reexport`. `@require` is also interesting.

- Random grid search
- Support Julia algos
- njobs>1
- Dataframes
- Feature extraction (eg. text features)
- More tests
- Review docstrings

More long-term
-----
- JLD/Pickling (possibly by implementing JLD serialization for PyObjects
through pickling)
- Sparse matrices (grep for 'sparse' in /test)
- Write KFold, StratifiedKFold, etc. to automatically account for Julian
  indices being 1-based