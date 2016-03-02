using MacroTools
using PyCall

# We don't strictly need this, but it's convenient for writing examples
include("Ndgrid.jl")

export @pyimport2

@pyimport types as py_types
ispymodule(obj::PyObject) = pyisinstance(mm, py_types.ModuleType)


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
    nuni = nunique([size(X, 1) for X in collect(filter(x->x!=nothing, arrays))])
    if nuni > 1
        throw(ArgumentError("Found arrays with inconsistent numbers of samples: $nuni"))
    end
end


""" Turns `Dict("x"=>10, "y"=>40)` into `Dict(:x=>10, :y=>40)` """
kwargify(assoc::Associative) =
    Dict([Symbol(k)=>v for (k, v) in assoc])

