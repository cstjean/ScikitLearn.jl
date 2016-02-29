# Adapted from https://github.com/MichaelHatherly/Docile.jl
# and http://maurow.bitbucket.org/notes/documenting-a-julia-package.html
# (C) Michael Hatherly, under MIT-license
using Lexicon, Sklearn

const api_directory = "api"
const modules = [Sklearn, Skcore]

cd(dirname(@__FILE__)) do
    # Run the doctests *before* we start to generate *any* documentation.
    ## for m in modules
    ##     failures = failed(doctest(m))
    ##     if !isempty(failures.results)
    ##         println("\nDoctests failed, aborting commit.\n")
    ##         display(failures)
    ##         exit(1) # Bail when doctests fail.
    ##     end
    ## end
    # also execute examples
    #include("../examples/ex1.jl")

    # Generate and save the contents of docstrings as markdown files.
    index  = Index()
    for mod in modules
        update!(index, save(joinpath(api_directory, "$(mod).md"), mod))
    end
    save(joinpath(api_directory, "index.md"), index; md_subheader = :category)

    # Add a reminder not to edit the generated files.
    open(joinpath(api_directory, "README.md"), "w") do f
        print(f, """
        Files in this directory are generated using the `build.jl` script. Make
        all changes to the originating docstrings/files rather than these ones.
        Documentation should *only* be built directly on the `master` branch.
        Source links would otherwise become unavailable should a branch be
        deleted from the `origin`. This means potential pull request authors
        *should not* run the build script when filing a PR.
        """)
    end

    info("Adding all documentation changes in $(api_directory) to this commit.")
    success(`git add $(api_directory)`) || exit(1)
end
