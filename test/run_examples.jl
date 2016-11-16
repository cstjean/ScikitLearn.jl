ex_dir = "../examples/"
for fname in readdir(ex_dir)
    path = ex_dir * fname
    if endswith(fname, ".ipynb")
        println("Testing $path")
        @eval module Testing
            using NBInclude
            nbinclude($path)
        end
    end
end
