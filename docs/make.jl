using Pkg
using Documenter
using ScikitLearn
using PyPlot #pre-installs matplotlib
import PyCall
# Install scikit-learn if not installed
PyCall.pyimport_conda("sklearn", "scikit-learn")


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
	repo = "github.com/cstjean/ScikitLearn.jl.git"
	)


