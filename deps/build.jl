import Conda
#Workaround the MacOS build issue when building the package with Conda 
@static if Sys.isapple()
    if !haskey(ENV, "PYTHON") || ENV["PYTHON"] == "" || ENV["PYTHON"] == "Conda"
      #use non-mkl versions of python packages
        Conda.add("nomkl")
        Conda.rm("mkl")
    end
end 
