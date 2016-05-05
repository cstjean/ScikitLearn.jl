# Adapted from scikit-learn
# Copyright (c) 2007–2016 The scikit-learn developers.

include("../src/Skcore.jl")  # a bit awkward - FIXME

include("test_base.jl")
all_test_base()

include("test_pipeline.jl")
all_test_pipeline()

include("test_crossvalidation.jl")
all_test_crossvalidation()

include("test_utils.jl")

nothing
