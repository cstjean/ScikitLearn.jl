using Conda
if Base.VERSION <= v"1.6.2" 
	# GLIBCXX_3.4.26 
	Conda.add("libstdcxx-ng>=3.4,<9.2", channel="conda-forge")
else 
	# GLIBCXX_3.4.29 
	# checked up to v1.8.0 
	Conda.add("libstdcxx-ng>=3.4,<11.4", channel="conda-forge")
end 

using Pkg
using Documenter
using ScikitLearn
using PyPlot #pre-installs matplotlib
import PyCall
PyCall.pyimport_conda("sklearn", "scikit-learn") #preinstalls scikit-learn


pages = [
	"Introduction" => "index.md",
	"Quick Start Guide" => "man/quickstart.md",
	"Model API"  => "man/api.md",
	"Reading the Scikit-learn Documentation" => "man/python.md",
	"Example Gallery" => "man/examples_refer.md",
	"Models" => "man/models.md",
	"Cross-validation" => "man/cross_validation.md",
	"Model Selection" => "man/model_selection.md",
	"Pipelines and FeatureUnion" => "man/pipelines.md",
	"DataFrames" => "man/dataframes.md",
	"Saving Models to Disk" => "man/jld.md",
	"Getting Help" => "man/help.md"
	]

for (page, link) in pages
    println("$page\t=>$link")
end


makedocs(
	sitename = "ScikitLearn.jl",
	pages    = pages,
	format   = Documenter.HTML(
			prettyurls = get(ENV, "CI", nothing) == "true"
			),
	)

deploydocs(
	repo   = "github.com/cstjean/ScikitLearn.jl.git"
	)


