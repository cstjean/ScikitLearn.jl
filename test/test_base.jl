using Base.Test
using Sklearn
using Sklearn.GridSearch: GridSearchCV
using Sklearn.Pipelines: Pipeline
using PyCall: PyError

@pyimport2 sklearn.svm: SVC
@pyimport2 sklearn.feature_selection: (SelectFpr, f_classif)


function test_is_classifier()
    svc = SVC()
    @test is_classifier(svc)
    @test is_classifier(GridSearchCV(svc, Dict("C"=>[0.1, 1])))
    @test is_classifier(Pipeline([("svc", svc)]))
    @test is_classifier(Pipeline([("svc_cv",
                                   GridSearchCV(svc, Dict("C"=>[0.1, 1])))]))
end


function test_set_params()
    # test nested estimator parameter setting
    clf = Pipeline([("svc", SVC())])
    # non-existing parameter in svc
    @test_throws(PyError, set_params!(clf, svc__stupid_param=true))
    # non-existing parameter of pipeline
    @test_throws(ArgumentError, set_params!(clf, svm__stupid_param=true))
end


function test_clone()
    # Tests that clone creates a correct deep copy.
    # We create an estimator, make a copy of its original state
    # (which, in this case, is the current state of the estimator),
    # and check that the obtained copy is a correct deep copy.

    selector = SelectFpr(f_classif, alpha=0.1)
    new_selector = clone(selector)
    @test(selector !== new_selector)
    @test(get_params(selector) == get_params(new_selector))

    selector = SelectFpr(f_classif, alpha=zeros(10, 2))
    new_selector = clone(selector)
    @test(selector !== new_selector)
end


test_is_classifier()
test_set_params()
test_clone()

:ok
