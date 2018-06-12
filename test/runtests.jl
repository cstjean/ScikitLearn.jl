# Adapted from scikit-learn
# Copyright (c) 2007–2016 The scikit-learn developers.

# Install scikit-learn if not installed
import PyCall
PyCall.pyimport_conda("sklearn", "scikit-learn")

using ScikitLearn
using Base.Test

@testset "ScikitLearnTests" begin

    @testset "models" begin
        include("test_models.jl") # runs tests automatically
    end

    @testset "base" begin
        include("test_base.jl")
        all_test_base()
    end

    @testset "pipeline" begin
        include("test_pipeline.jl")
        all_test_pipeline()
    end

    @testset "crossvalidation" begin
        include("test_crossvalidation.jl")
        all_test_crossvalidation()
    end

    @testset "utils" begin
        include("test_utils.jl")
    end
    @testset "quickstart" begin
        include("test_quickstart.jl")
    end
    @testset "DataFrames" begin
        include("test_dataframes.jl")
    end

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

    @testset "Notebook examples" begin
        run_examples()
    end

end

nothing
