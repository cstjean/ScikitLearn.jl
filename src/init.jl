
#This ensures this is done once only when the module is first loaded

function __init__()
    
    mod = PyCall.pyimport_conda("sklearn", "scikit-learn")
    version = VersionParsing.vparse(mod.__version__)
    min_version = v"0.18.0"
    if version < min_version
        @warn("Your Python's scikit-learn has version $version. We recommend updating to $min_version or higher for best compatibility with ScikitLearn.jl.")
    end
end


