*Credits: this code and documentation was adapted from Paul Butler's [sklearn-pandas](https://github.com/paulgb/sklearn-pandas)*

# DataFrames

It is possible to use a [dataframe](https://github.com/JuliaStats/DataFrames.jl) as a training set, but it needs to be converted to an array first. `DataFrameMapper` is used to specify how this conversion proceeds. For example, PCA might be applied to some numerical dataframe columns, and one-hot-encoding to a categorical column.

## Transformation Mapping

Consider this dataset:


```jldoctest dataframes
julia> using ScikitLearn

julia> using DataFrames: DataFrame, missing

julia> @sk_import preprocessing: (LabelBinarizer, StandardScaler)
PyObject <class 'sklearn.preprocessing._data.StandardScaler'>

julia> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])
8×3 DataFrames.DataFrame
│ Row │ pet    │ children │ salary │
│     │ String │ Float64  │ Int64  │
├─────┼────────┼──────────┼────────┤
│ 1   │ cat    │ 4.0      │ 90     │
│ 2   │ dog    │ 6.0      │ 24     │
│ 3   │ dog    │ 3.0      │ 44     │
│ 4   │ fish   │ 3.0      │ 27     │
│ 5   │ cat    │ 2.0      │ 32     │
│ 6   │ dog    │ 3.0      │ 59     │
│ 7   │ cat    │ 5.0      │ 36     │
│ 8   │ fish   │ 4.0      │ 27     │


```

### Map the Columns to Transformations

The mapper takes a list of pairs. The first is a column name from the DataFrame, or a list containing one or multiple columns (we will see an example with multiple columns later). The second is an object which will perform the transformation which will be applied to that column:

Note: `ScikitLearn.DataFrameMapper` won't be available until `DataFrames` is imported


```jldoctest dataframes
julia> mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())]);
```

The difference between specifying the column selector as `:column` (as a single symbol) and `[:column]` (as a list with one element) is the shape of the array that is passed to the transformer. In the first case, a one dimensional array with be passed, while in the second case it will be a 2-dimensional array with one column, i.e. a column vector.

### Test the Transformation

We can use the `fit_transform!` shortcut to both fit the model and see what transformed data looks like. In this and the other examples, output is rounded to two digits with `round` to account for rounding errors on different hardware:


```jldoctest dataframes
julia> round.(fit_transform!(mapper, copy(data)), digits=2)
8×4 Array{Float64,2}:
 1.0  0.0  0.0   0.21
 0.0  1.0  0.0   1.88
 0.0  1.0  0.0  -0.63
 0.0  0.0  1.0  -0.63
 1.0  0.0  0.0  -1.46
 0.0  1.0  0.0  -0.63
 1.0  0.0  0.0   1.04
 0.0  0.0  1.0   0.21

```

Note that the first three columns are the output of the LabelBinarizer (corresponding to `cat`, `dog`, and `fish`
 respectively) and the fourth column is the standardized value for the number of children. In general, the columns are ordered according to the order given when the DataFrameMapper is constructed.

Now that the transformation is trained, we confirm that it works on new data:


```jldoctest dataframes
julia> sample = DataFrame(pet = ["cat"], children = [5.])
1×2 DataFrames.DataFrame
│ Row │ pet    │ children │
│     │ String │ Float64  │
├─────┼────────┼──────────┤
│ 1   │ cat    │ 5.0      │

julia> round.(transform(mapper, sample), digits=2)
1×4 Array{Float64,2}:
 1.0  0.0  0.0  1.04

```

### Transform Multiple Columns

Transformations may require multiple input columns. In these cases, the column names can be specified in a list:


```jldoctest dataframes
julia> @sk_import decomposition: PCA
PyObject <class 'sklearn.decomposition._pca.PCA'>

julia> mapper2 = DataFrameMapper([([:children, :salary], PCA(1))]);

```

Now running `fit_transform!` will run PCA on the `children` and `salary` columns and return the first principal component:


```jldoctest dataframes
julia> round.(fit_transform!(mapper2, copy(data)), digits=1)
8×1 Array{Float64,2}:
  47.6
 -18.4
   1.6
 -15.4
 -10.4
  16.6
  -6.4
 -15.4

```

### Multiple transformers for the same column

Multiple transformers can be applied to the same column specifying them in a list:


```jldoctest dataframes
julia> @sk_import impute: SimpleImputer
PyObject <class 'sklearn.impute._base.SimpleImputer'>

julia> mapper3 = DataFrameMapper([([:age], [SimpleImputer(),
                                     StandardScaler()])]; missing2NaN=true);

julia> data_3 = DataFrame(age= [1, missing, 3]);

julia> fit_transform!(mapper3, data_3)
3×1 Array{Float64,2}:
 -1.224744871391589
  0.0
  1.224744871391589

```

### Columns that don't need any transformation

Only columns that are listed in the `DataFrameMapper` are kept. To keep a column but don't apply any transformation to it, use `nothing` as transformer:


```jldoctest dataframes
julia> mapper3 = DataFrameMapper([(:pet, LabelBinarizer()), (:children, nothing)]);

julia> round.(fit_transform!(mapper3, copy(data)))
8×4 Array{Float64,2}:
 1.0  0.0  0.0  4.0
 0.0  1.0  0.0  6.0
 0.0  1.0  0.0  3.0
 0.0  0.0  1.0  3.0
 1.0  0.0  0.0  2.0
 0.0  1.0  0.0  3.0
 1.0  0.0  0.0  5.0
 0.0  0.0  1.0  4.0

```

## Cross-validation

Now that we can combine features from a DataFrame, we may want to use cross-validation to see whether our model works.


```jldoctest dataframes
julia> @sk_import linear_model: LinearRegression
PyObject <class 'sklearn.linear_model._base.LinearRegression'>

julia> using ScikitLearn.CrossValidation: cross_val_score

julia> pipe = Pipelines.Pipeline([(:featurize, mapper), (:lm, LinearRegression())]);

julia> round.(cross_val_score(pipe, data, data[!,:salary]), digits=2)
3-element Array{Float64,1}:
  -1.09
  -5.3
 -15.38

```
