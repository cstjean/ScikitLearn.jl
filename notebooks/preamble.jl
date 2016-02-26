push!(LOAD_PATH, "/Users/cedric/Programa/Sklearn/src")
push!(LOAD_PATH, "/Users/cedric/Programa/Mario")
using Autoreload

arequire("Skcore")
using Skcore

include("../src/Ndgrid.jl")

using PyCall
using PyPlot
