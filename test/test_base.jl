using Base.Test
using Sklearn
using Sklearn.GridSearch: GridSearchCV
using Sklearn.Pipelines: Pipeline

@pyimport2 sklearn.svm: SVC


function test_is_classifier()
    svc = SVC()
    @test is_classifier(svc)
    @test is_classifier(GridSearchCV(svc, Dict("C"=>[0.1, 1])))
    @test is_classifier(Pipeline([("svc", svc)]))
    @test is_classifier(Pipeline([("svc_cv",
                                   GridSearchCV(svc, Dict("C"=>[0.1, 1])))]))
end


test_is_classifier()
