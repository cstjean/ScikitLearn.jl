using MacroTools
using PyCall

# We don't strictly need this, but it's convenient for writing examples
include("Ndgrid.jl")

export @pyimport2

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
        gensyms = [gensym() for _ in members]
        expansion(m, g) =
            begin
                ## @show :(PyCall.@pyimport $(esc(Expr(:., mod, QuoteNode(m)))) as $(esc(g)))
                mname = PyCall.modulename(m)
            :(try
                # If it's a module
                global const $(esc(m)) = PyCall.pywrap(PyCall.pyimport($mname))
                #PyCall.@pyimport $(esc(Expr(:., mod, Expr(:quote, m)))) PyCall.as $(esc(g))
                #$(esc(:(global $m = $g)))
            catch $(esc(:e))
                # If it's a variable/function
                if $(esc(:(isa(e, PyCall.PyError))))
                    $(esc(:(PyCall.@pyimport $mod as $g)))
                    $(esc(:(global $m = $g.$m)))
                else
                    rethrow()
                end
            end) end
        # This is a bad expansion (putting everything in `esc`). FIXME
        :(begin
            $([expansion(m, g) for (m, g) in zip(members, gensyms)]...)
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

