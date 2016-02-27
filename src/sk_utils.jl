using MacroTools
using PyCall

export @pyimport2, check_consistent_length


"""
    @pyimport2 sklearn: (decomposition, clone)

is the same as the Python code:

from sklearn import decomposition, clone

"""
macro pyimport2(expr)
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
            :(try
                # If it's a module
                PyCall.@pyimport $mod.$m as $g
                global $m = $g
            catch e
                # If it's a variable/function
                if isa(e, PyCall.PyError)
                    PyCall.@pyimport $mod as $g
                    global $m = $g.$m
                else
                    rethrow()
                end
            end)
        # This is a bad expansion (putting everything in `esc`). FIXME
        esc(:(begin
            using PyCall: @pyimport
            $([expansion(m, g) for (m, g) in zip(members, gensyms)]...)
            end))
    else
        esc(:(PyCall.@pyimport $expr))
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


""" `parse_function_definition(fdef)`

Macro helper: parses the given function definition and returns
`(fname::Symbol, args::Vector{Any}, kwargs::Vector{Any}, body::Vector{Any})`

One can rebuild the function definition with
`esc(:(function &fname (&(args...); &(kwargs...)) &(body...) end))`

(replace the & with dollar sign - the docstring system didn't allow us to write
it correctly) """
function parse_function_definition(fdef)
    if @capture(fdef, function fname_(args__; kwargs__) body__ end)
        (fname, args, kwargs, body)
    elseif @capture(fdef, function fname_(args__) body__ end)
        (fname, args, Any[], body)
    elseif @capture(fdef, (fname_(args__; kwargs__) = bexpr_))
        (fname, args, kwargs, Any[bexpr])
    elseif @capture(fdef, fname_(args__) = bexpr_)
        (fname, args, Any[], Any[bexpr])
    else
        error("Not a function definition: $fdef")
    end
end

""" `macro_keyword_args(kwarg)`

Macro helper: if a function/macro definition contains x=y in an argument list,
this will return (x, y) """ # TODO: add example
function parse_kwarg(kwarg)
    @assert kwarg.head == :kw
    return (kwarg.args[1], kwarg.args[2])
end
