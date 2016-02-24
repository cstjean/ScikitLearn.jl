using MacroTools


"""
    @pyimport2 sklearn:(decomposition, clone)

is the same as the Python code:

import sklearn: decomposition, clone

"""
macro pyimport2(expr)
    if @capture(expr, mod_:what_)
        if isa(what, Symbol)
            members = [what]
        else
            @assert @capture(what, ((members__),)) "Bad @pyimport2 statement"
        end
        gensyms = [gensym() for _ in members]
        m = members[1]
        g = gensyms[1]
        expansion(m, g) =
           :(try
                @pyimport $mod.$m as $g
                global $m = $g
            catch e
                if isa(e, PyCall.PyError)
                    @pyimport $mod as $g
                    global $m = $g.$m
                else
                    rethrow()
                end
            end)
        esc(:(begin $([expansion(m, g) for (m, g) in zip(members, gensyms)]...) end))
    else
        :(@pyimport $expr)
    end
end
