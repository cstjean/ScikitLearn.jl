import Conda
#Workaround the MacOS build issue when building the package with Conda 

 if !haskey(ENV, "PYTHON") || ENV["PYTHON"] == "" || ENV["PYTHON"] == "Conda"
    if Sys.isapple()  
      #use non-mkl versions of python packages
        Conda.add("nomkl")
        Conda.rm("mkl")
        #installs scikit-learn if not already installed
        PyCall.pyimport_conda("sklearn", "scikit-learn") 
    else
      #installs scikit-learn if not already installed
        PyCall.pyimport_conda("sklearn", "scikit-learn")
    end
end 
