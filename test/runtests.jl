# Adapted from scikit-learn
# Copyright (c) 2007–2016 The scikit-learn developers.

# Install scikit-learn if not installed
import PyCall
PyCall.pyimport_conda("sklearn", "scikit-learn")

using ScikitLearn

include("test_models.jl") # runs tests automatically

include("test_base.jl")
all_test_base()

include("test_pipeline.jl")
all_test_pipeline()

include("test_crossvalidation.jl")
all_test_crossvalidation()

include("test_utils.jl")
include("test_quickstart.jl")
include("test_dataframes.jl")

exceptions = ["Density_Estimation_Julia.ipynb", #GaussianMixtures fails on 0.6 as of MAR17
              "Plot_Kmeans_Digits_Julia.ipynb", # LowRankModels fails on 0.6 as of MAR17
              "Simple_1D_Kernel_Density.ipynb", # https://github.com/JuliaPy/PyCall.jl/issues/372
              ]
function run_examples()
    ex_dir = "../examples/"
    for fname in readdir(ex_dir)
        if !(fname in exceptions)
            path = ex_dir * fname
            if endswith(fname, ".ipynb")
                println("Testing $path")
                @eval module Testing
                    using NBInclude
                    nbinclude($path)
                end
            end
        end
    end
end

run_examples()

nothing
