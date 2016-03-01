# Adapted from scikit-learn
# Copyright (c) 2007–2016 The scikit-learn developers.

using Base.Test
using Skcore
using Skcore: GridSearchCV
using Skcore: Pipeline, is_classifier
using PyCall: PyError

@pyimport2 sklearn.svm: SVC
@pyimport2 sklearn.feature_selection: (SelectFpr, f_classif)
@pyimport2 sklearn.tree: (DecisionTreeClassifier, DecisionTreeRegressor)
@pyimport2 sklearn: datasets


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


function test_score_sample_weight()
    srand(0)

    # test both ClassifierMixin and RegressorMixin
    estimators = [DecisionTreeClassifier(max_depth=2),
                  DecisionTreeRegressor(max_depth=2)]
    sets = Any[datasets.load_iris(),
               datasets.load_boston()]

    for (est, ds) in zip(estimators, sets)
        fit!(est, ds["data"], ds["target"])
        # generate random sample weights
        sample_weight = rand(1:10, length(ds["target"]))
        # check that the score with and without sample weights are different
        @test (score(est, ds["data"], ds["target"]) !=
               score(est, ds["data"], ds["target"],
                     sample_weight=sample_weight))
    end
end


function all_test_base()
    test_is_classifier()
    test_set_params()
    test_clone()
    test_score_sample_weight()
end


:ok
