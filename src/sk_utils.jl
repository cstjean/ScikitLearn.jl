using MacroTools
using PyCall

export @pyimport2


"""
    @pyimport2 sklearn: (decomposition, clone)

is the same as the Python code:

from sklearn import decomposition, clone

"""
macro pyimport2(expr)
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
                @pyimport $mod.$m as $g
                global $m = $g
            catch e
                # If it's a variable/function
                if isa(e, PyCall.PyError)
                    @pyimport $mod as $g
                    global $m = $g.$m
                else
                    rethrow()
                end
            end)
        esc(:(begin $([expansion(m, g) for (m, g) in zip(members, gensyms)]...)
            end))
    else
        esc(:(@pyimport $expr))
    end
end


nunique(iter) = length(Set(iter)) # slow definition
