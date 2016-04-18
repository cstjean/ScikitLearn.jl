# NOTE: Most of the codebase is in the module Skcore. ScikitLearn is
# defined here by importing from Skcore and reexporting what we want in the
# interface. This arrangement simplifies the codebase and allowed us to
# experiment with different submodule structures without breaking everything.
# TODO: get rid of Skcore

module ScikitLearn

include("Skcore.jl")

using PyCall: @pyimport
importall ScikitLearnBase
using ScikitLearn.Skcore: @sk_import

export @sk_import, CrossValidation, Pipelines, GridSearch


################################################################################

"""    @reexportsk(identifiers...)
is equivalent to
    using Skcore: identifiers...
    export identifiers...
"""
macro reexportsk(identifiers...)
    esc(:(begin
        $([:(begin
             using ScikitLearn.Skcore: $idf
             export $idf
             end)
        for idf in identifiers]...)
    end))
end

module CrossValidation
using ..@reexportsk, ..@sk_import
using ScikitLearn.Skcore: @pyimport2
@reexportsk(cross_val_score, cross_val_predict)
@pyimport2 sklearn.cross_validation: train_test_split
export train_test_split

using ..Skcore: cv_iterator_syms

@eval @reexportsk($(cv_iterator_syms...))
end


module Pipelines
using ..@reexportsk
@reexportsk(Pipeline, make_pipeline, FeatureUnion, named_steps)
end


module GridSearch
using ..@reexportsk
@reexportsk(GridSearchCV, RandomizedSearchCV)
end


module Utils
using ..@reexportsk
using ScikitLearn.Skcore: @pyimport2
@reexportsk(meshgrid)
export @pyimport2
end


module Preprocessing
# These are my own extensions. I'd like to keep ScikitLearn close to
# scikit-learn, which is why they are not exported so far. I might move them
# elsewhere. - cstjean
include("dictencoder.jl")
include("preprocessing.jl")
end


using Requires
@require DataFrames include("dataframes.jl")

include("models/models.jl")

################################################################################
# Other exports

# Not sure if we should export all the api. set_params!/get_params are rarely
# used by user code.
for f in ScikitLearnBase.api @eval(@reexportsk $f) end

end
