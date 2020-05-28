import Conda
#Workaround the MacOS build issue when building the package with Conda 
@static if Sys.isapple()
    if !haskey(ENV, "PYTHON") || ENV["PYTHON"] == "" || lowercase(ENV["PYTHON"]) == "conda"
      #use non-mkl versions of python packages
        Conda.add("nomkl")
        Conda.rm("mkl")
    end
end 
