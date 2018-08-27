# Code from http://scikitlearnjl.readthedocs.io/en/latest/dataframes/

module TestingDataFrames

using ScikitLearn
using DataFrames: DataFrame, missing
@sk_import preprocessing: (LabelBinarizer, StandardScaler)

data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])

mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())]);

round.(fit_transform!(mapper, copy(data)), digits=2)

sample = DataFrame(pet= ["cat"], children= [5.])
round.(transform(mapper, sample), digits=2)

@sk_import decomposition: PCA
mapper2 = DataFrameMapper([([:children, :salary], PCA(1))]);

round.(fit_transform!(mapper2, copy(data)), digits=1)


@sk_import preprocessing: (Imputer, StandardScaler)
mapper3 = DataFrameMapper([([:age], [Imputer(),
                                     StandardScaler()])]; missing2NaN=true)
data_3 = DataFrame(age= [1, missing, 3])
fit_transform!(mapper3, data_3)


mapper3 = DataFrameMapper([
     (:pet, LabelBinarizer()),
     (:children, nothing)])
round.(fit_transform!(mapper3, copy(data)))


@sk_import linear_model: LinearRegression
using ScikitLearn.CrossValidation: cross_val_score

pipe = Pipelines.Pipeline([
     (:featurize, mapper),
     (:lm, LinearRegression())])
round.(cross_val_score(pipe, data, data[:salary]), digits=2)

end
