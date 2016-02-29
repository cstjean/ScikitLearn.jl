# All code is in Skcore. Sklearn is defined here by importing from Skcore and
# reexporting what we want. This arrangement simplifies the codebase and allows
# us to experiment with different submodule structures without breaking
# everything.
include("Skcore.jl")


module Sklearn

import Skcore

using PyCall: @pyimport
using SklearnBase
using SklearnBase: @import_api, @simple_estimator_constructor

export LinearModels, CrossValidation, Datasets, Ensembles, Trees

################################################################################

macro reexport(module_, identifiers...)
    esc(:(begin
        $([:(begin
             using $module_: $idf
             export $idf
             end)
        for idf in identifiers]...)
    end))
end

"""    reexportsk(identifiers...)
is equivalent to
    using Skcore.identifiers...
    export identifiers...
"""
macro reexportsk(identifiers...)
    :(@reexport($(esc(:Skcore)), $(map(esc, identifiers)...)))
end


module LinearModels
using ..@reexport
using PyCall: @pyimport
@pyimport sklearn.linear_model as _linear_model
for var in names(_linear_model)
    if isa(var, Symbol) && string(var)[1] != '_'
        @eval const $var = _linear_model.$var
    end
end
end


module CrossValidation
using ..@reexportsk
@reexportsk(cross_val_score)
end


module Pipelines
using ..@reexportsk
@reexportsk(Pipeline, make_pipeline, FeatureUnion)
end


@pyimport sklearn.ensemble as Ensembles
@pyimport sklearn.datasets as Datasets
@pyimport sklearn.tree as Trees


################################################################################
# Other exports

# Not sure if we should export all the api
for f in SklearnBase.api @eval(@reexportsk $f) end



end
