
Credits: this code and documentation was adapted from Paul Butler's [sklearn-pandas](https://github.com/paulgb/sklearn-pandas)

# DataFrames

It is possible to train models based on dataframes, but they need to be converted to arrays first. `DataFrameMapper` is used to specify how this conversion proceeds. For example, PCA might be applied to some numerical dataframe columns, and one-hot-encoding to a categorical column.

## Transformation Mapping


```julia
using ScikitLearn
using DataFrames: DataFrame, NA, DataArray
using DataArrays: @data
@sk_import preprocessing: (LabelBinarizer, StandardScaler)

data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])
```




<table class="data-frame"><tr><th></th><th>pet</th><th>children</th><th>salary</th></tr><tr><th>1</th><td>cat</td><td>4.0</td><td>90</td></tr><tr><th>2</th><td>dog</td><td>6.0</td><td>24</td></tr><tr><th>3</th><td>dog</td><td>3.0</td><td>44</td></tr><tr><th>4</th><td>fish</td><td>3.0</td><td>27</td></tr><tr><th>5</th><td>cat</td><td>2.0</td><td>32</td></tr><tr><th>6</th><td>dog</td><td>3.0</td><td>59</td></tr><tr><th>7</th><td>cat</td><td>5.0</td><td>36</td></tr><tr><th>8</th><td>fish</td><td>4.0</td><td>27</td></tr></table>



### Map the Columns to Transformations

The mapper takes a list of pairs. The first is a column name from the DataFrame, or a list containing one or multiple columns (we will see an example with multiple columns later). The second is an object which will perform the transformation which will be applied to that column:

Note: `ScikitLearn.DataFrameMapper` won't be available until `DataFrames` is imported


```julia
mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())]);
```



The difference between specifying the column selector as :column (as a single symbol) and [:column] (as a list with one element) is the shape of the array that is passed to the transformer. In the first case, a one dimensional array with be passed, while in the second case it will be a 2-dimensional array with one column, i.e. a column vector.

### Test the Transformation

We can use the `fit_transform!` shortcut to both fit the model and see what transformed data looks like. In this and the other examples, output is rounded to two digits with `round` to account for rounding errors on different hardware:


```julia
round(fit_transform!(mapper, copy(data)), 2)
```




    8x4 Array{Float64,2}:
     1.0  0.0  0.0   0.21
     0.0  1.0  0.0   1.88
     0.0  1.0  0.0  -0.63
     0.0  0.0  1.0  -0.63
     1.0  0.0  0.0  -1.46
     0.0  1.0  0.0  -0.63
     1.0  0.0  0.0   1.04
     0.0  0.0  1.0   0.21



Note that the first three columns are the output of the LabelBinarizer (corresponding to `cat`, `dog`, and `fish`
 respectively) and the fourth column is the standardized value for the number of children. In general, the columns are ordered according to the order given when the DataFrameMapper is constructed.

Now that the transformation is trained, we confirm that it works on new data:


```julia
sample = DataFrame(pet= ["cat"], children= [5.])
round(transform(mapper, sample), 2)
```




    1x4 Array{Float64,2}:
     1.0  0.0  0.0  1.04



### Transform Multiple Columns

Transformations may require multiple input columns. In these cases, the column names can be specified in a list:


```julia
@sk_import decomposition: PCA
mapper2 = DataFrameMapper([([:children, :salary], PCA(1))]);
```

Now running `fit_transform!` will run PCA on the `children` and `salary` columns and return the first principal component:


```julia
round(fit_transform!(mapper2, copy(data)), 1)
```




    8x1 Array{Float64,2}:
      47.6
     -18.4
       1.6
     -15.4
     -10.4
      16.6
      -6.4
     -15.4



### Multiple transformers for the same column

Multiple transformers can be applied to the same column specifying them in a list:


```julia
@sk_import preprocessing: Imputer
mapper3 = DataFrameMapper([([:age], [Imputer()])]; NA2NaN=true)
data_3 = DataFrame(age= @data([1, NA, 3]))
fit_transform!(mapper3, data_3)
```




    3x1 Array{Float64,2}:
     1.0
     2.0
     3.0



### Columns that don't need any transformation

Only columns that are listed in the `DataFrameMapper` are kept. To keep a column but don't apply any transformation to it, use `nothing` as transformer:


```julia
mapper3 = DataFrameMapper([
     (:pet, LabelBinarizer()),
     (:children, nothing)])
round(fit_transform!(mapper3, copy(data)))
```




    8x4 Array{Float64,2}:
     1.0  0.0  0.0  4.0
     0.0  1.0  0.0  6.0
     0.0  1.0  0.0  3.0
     0.0  0.0  1.0  3.0
     1.0  0.0  0.0  2.0
     0.0  1.0  0.0  3.0
     1.0  0.0  0.0  5.0
     0.0  0.0  1.0  4.0



## Cross-validation

Now that we can combine features from a DataFrame, we may want to use cross-validation to see whether our model works.


```julia
@sk_import linear_model: LinearRegression

pipe = Pipelines.Pipeline([
     (:featurize, mapper),
     (:lm, LinearRegression())])
round(CrossValidation.cross_val_score(pipe, data, data[:salary]), 2)
```




    3-element Array{Float64,1}:
      -1.09
      -5.3 
     -15.38


