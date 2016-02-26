using Base.Test
using Skcore
using Skcore: Pipeline
using PyCall: PyError

@pyimport2 sklearn.svm: SVC
@pyimport2 sklearn.feature_selection: (SelectKBest, f_classif)

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)


"""Small class to test parameter dispatching.
"""
type T
    a
    b
    T(a=nothing, b=nothing) = new(a, b)
end


Skcore.fit!(self::T, X, y) = self

Skcore.get_params(self::T; deep=false) = Dict("a"=>self.a, "b"=>self.b)
function Skcore.set_params!(self; a=error("Missing value"))
    self.a = params[:a]
    return self
end


## class FitParamT(object):
##     """Mock classifier
##     """

##     def __init__(self):
##         self.successful = False
##         pass

##     def fit(self, X, y, should_succeed=False):
##         self.successful = should_succeed

##     def predict(self, X):
##         return self.successful


function test_pipeline_init()
    # Test the various init parameters of the pipeline.
    @test_throws(MethodError, Pipeline())

    # Check that we can't instantiate pipelines with objects without fit
    # method
    # Julia TODO
    ## @test_throws(TypeError, Pipeline, [('svc', IncorrectT)])

    # Smoke test with only an estimator
    clf = T()
    pipe = Pipeline([("svc", clf)])
    @test (get_params(pipe, deep=true) ==
           Dict("svc__a"=>nothing, "svc__b"=>nothing, "svc"=>clf))

    # Check that params are set
    set_params!(pipe; svc__a=0.1)
    @test clf.a==0.1
    @test clf.b==nothing
    # Smoke test the repr:
    #repr(pipe) Julia TODO

    # Test with two objects
    clf = SVC()
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([("anova", filter1), ("svc", clf)])

    # Check that we can"t use the same stage name twice. Julia TODO
    # @test_throws(ValueError, Pipeline, [("svc", SVC()), ("svc", SVC())])

    # Check that params are set
    set_params!(pipe, svc__C=0.1)
    @test clf[:C]==0.1
    # Smoke test the repr: Julia TODO
    # repr(pipe)

    # Check that params are not set when naming them wrong
    #assert_raises(ValueError, pipe.set_params, anova__C=0.1)

    ## # Test clone
    ## pipe2 = clone(pipe)
    ## assert_false(pipe.named_steps["svc"] is pipe2.named_steps["svc"])

    ## # Check that apart from estimators, the parameters are the same
    ## params = pipe.get_params()
    ## params2 = pipe2.get_params()
    ## # Remove estimators that where copied
    ## params.pop("svc")
    ## params.pop("anova")
    ## params2.pop("svc")
    ## params2.pop("anova")
    ## assert_equal(params, params2)
end
