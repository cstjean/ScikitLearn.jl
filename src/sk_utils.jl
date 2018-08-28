using MacroTools
using MacroTools: @forward
using PyCall
import Compat

# We don't strictly need this, but it's convenient for writing examples
include("Ndgrid.jl")

export @pyimport2, type_of_target

# TODO: this should be in PyCall.jl
"""
    @pyimport2 sklearn: (decomposition, clone)

is the same as the Python code:

    from sklearn import decomposition, clone

"""
macro pyimport2(expr, optional_varname...)
    if isa(expr, Expr)
        @assert expr.head != :tuple "@pyimport2 requires parentheses, eg. @pyimport2 module_name: (a, b)"
    end
    if @capture(expr, mod_:what_)
        if isa(what, Symbol)
            members = [what]
        else
            @assert @capture(what, ((members__),)) "Bad @pyimport2 statement"
        end
        # I hate importing the Python objects at macro-expansion time, but I
        # couldn't figure out how to write a proper macro version of this. TODO
        module_ = PyCall.pyimport(PyCall.modulename(mod))
        function get_obj(member)
            try
                return pywrap(PyCall.pyimport(PyCall.modulename(mod) * "." *
                                              string(member)))
            catch e
                if isa(e, PyCall.PyError)
                    return module_[member]
                else rethrow() end
            end
        end
        py_objects = map(get_obj, members)
        :(begin
            $([quote
               if !isdefined($(Expr(:quote, member)))
                   const $(esc(member)) = $obj
               elseif !isa($(esc(member)), Union{Module, PyObject})
                   error("@pyimport2: ", $(Expr(:quote, member)),
                         " already defined")
               end
               end
               for (member, obj) in zip(members, py_objects)]...)
            nothing
            end)
    else
        :(@pyimport($(esc(expr)), $(map(esc, optional_varname)...)))
    end
end

nunique(iter) = length(Set(iter)) # slow definition


"""Check that all arrays have consistent first dimensions.

Checks whether all objects in arrays have the same shape or length.

Parameters
----------
*arrays : list or tuple of input objects.
    Objects that will be checked for consistent length.
"""
function check_consistent_length(arrays...)
    nuni = nunique([size(X, 1) for X in filter(x->x!=nothing, collect(arrays))])
    if nuni > 1
        throw(ArgumentError("Found arrays with inconsistent numbers of samples: $nuni"))
    end
end


""" Turns `Dict("x"=>10, "y"=>40)` into `Dict(:x=>10, :y=>40)` """
kwargify(assoc::AbstractDict) =
    Dict{Symbol, Any}([Pair(Symbol(k),v) for (k, v) in assoc])

""" Check if ``y`` is in a multilabel format.

Parameters
----------
y : numpy array of shape [n_samples]
    Target values.

Returns
-------
out : bool,
    Return ``True``, if ``y`` is in a multilabel format, else ```False``.

Examples
--------
>>> import numpy as np
>>> from sklearn.utils.multiclass import is_multilabel
>>> is_multilabel([0, 1, 0, 1])
False
>>> is_multilabel([[1], [0, 2], []])
False
>>> is_multilabel(np.array([[1, 0], [0, 0]]))
True
>>> is_multilabel(np.array([[1], [0], [0]]))
False
>>> is_multilabel(np.array([[1, 0, 0]]))
True
"""
function is_multilabel(y::AbstractArray{T}) where T
    if !(ndims(y) == 2 && size(y, 2) > 1)
        return false
    end

    if issparse(y)
        TODO()
        ## if isinstance(y, (dok_matrix, lil_matrix)):
        ##     y = y.tocsr()
        ## return (len(y.data) == 0 or np.unique(y.data).size == 1 and
        ##         (y.dtype.kind in 'biu' or  # bool, int, uint
        ##          _is_integral_float(np.unique(y.data))))
    else
        labels = unique(y)

        return length(labels) < 3 && (T <: Union{Bool, Integer})
    end
end


"""Determine the type of data indicated by target `y`

Parameters
----------
y : array-like

Returns
-------
target_type : string
    One of:
    * 'continuous': `y` is an array-like of floats that are not all
      integers, and is 1d or a column vector.
    * 'continuous-multioutput': `y` is a 2d array of floats that are
      not all integers, and both dimensions are of size > 1.
    * 'binary': `y` contains <= 2 discrete values and is 1d or a column
      vector.
    * 'multiclass': `y` contains more than two discrete values, is not a
      sequence of sequences, and is 1d or a column vector.
    * 'multiclass-multioutput': `y` is a 2d array that contains more
      than two discrete values, is not a sequence of sequences, and both
      dimensions are of size > 1.
    * 'multilabel-indicator': `y` is a label indicator matrix, an array
      of two dimensions with at least two columns, and at most 2 unique
      values.
    * 'unknown': `y` is array-like but none of the above, such as a 3d
      array, sequence of sequences, or an array of non-sequence objects.

Examples
--------
type_of_target([0.1, 0.6]) == "continuous"
type_of_target([1, -1, -1, 1]) == "binary"
type_of_target(["a", "b", "a"]) == "binary"
type_of_target([1.0, 2.0]) == "continuous"
type_of_target([1, 0, 2]) == "multiclass"
type_of_target([1.0, 0.0, 3.0]) == "continuous"
type_of_target(["a", "b", "c"]) == "multiclass"
type_of_target([1 2; 3 1]) == "multiclass-multioutput"
type_of_target([1.5 2.0; 3.0 1.6]) == "continuous-multioutput"
type_of_target([0 1; 1 1]) == "multilabel-indicator"
"""
function type_of_target(y::AbstractArray{T}) where T
    # Julia note: various compromises were made in translating this function
    if is_multilabel(y)
        return "multilabel-indicator"
    end

    if ndims(y) > 2
        return "unknown"
    end

    if ndims(y) == 2 && size(y, 2) == 0
        return "unknown"
    end

    if ndims(y) == 2 && size(y, 2) > 1
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]
    end

    # Julia note: Python used to check whether all elements were "round floats".
    # Julia is more strongly typed than Python, so I don't do that.
    if T <: AbstractFloat
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]]
        return "continuous" * suffix
    end

    if (length(unique(y)) > 2) || (ndims(y) >= 2 && size(y, 2) > 1)
        return "multiclass" * suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    else
        return "binary"  # [1, 2] or [["a"], ["b"]]
    end
end    

################################################################################

""" `FitBit(model)` will behave just like `model`, but also supports
`isfit(fb)`, which returns true IFF `fit!(model, ...)` has been called """
mutable struct FitBit
    model
    isfit::Bool
    FitBit(model) = new(model, false)
end
clone(fb::FitBit) = FitBit(clone(fb.model))

@forward FitBit.model transform, predict, predict_proba, predict_dist, get_classes

function fit!(fb::FitBit, args...; kwargs...)
    fit!(fb.model, args...; kwargs...)
    fb.isfit = true
    fb
end

isfit(fb::FitBit) = fb.isfit


################################################################################

function bincount(arr::AbstractVector{Int}; minlength=0)
    n = max(minlength, maximum(arr))
    out = zeros(Int, n)
    for x in arr
        if x < 1
            error("bincount first argument must be greater than or equal to 1")
        end
        out[x] += 1
    end
    out
end
