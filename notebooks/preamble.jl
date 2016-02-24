push!(LOAD_PATH, "/Users/cedric/Programa/Sklearn/src")
using Autoreload

arequire("Sklearn")
using Sklearn

include("../src/Ndgrid.jl")

using PyCall
using PyPlot
