# Adapted from scikit-learn
# Copyright (c) 2007–2016 The scikit-learn developers.

using Base.Test
importall ScikitLearnBase
using ScikitLearn
using ScikitLearn.Pipelines: Pipeline, FeatureUnion, make_pipeline
using ScikitLearnBase: @declare_hyperparameters
using ScikitLearn.Skcore: named_steps
using PyCall: PyError

@sk_import svm: SVC
@sk_import feature_selection: (SelectKBest, f_classif)
@sk_import datasets: load_iris
@sk_import linear_model: (LogisticRegression, LinearRegression)
@sk_import decomposition: (PCA, RandomizedPCA, TruncatedSVD)
@sk_import preprocessing: StandardScaler


JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)

############################################################
# Note: I defined T as TransfT because the two classes were nearly identical
# in the Python version

"""Small class to test parameter dispatching.
"""
type T
    a
    b
    T(a=nothing, b=nothing) = new(a, b)
end

fit!(self::T, X, y) = self
get_params(self::T; deep=false) = Dict("a"=>self.a, "b"=>self.b)
function set_params!(self::T; a=error("Missing value"))
    self.a = a
    return self
end
transform(self::T, X) = X


############################################################

""" Mock classifier """
type FitParamT
    successful
    FitParamT(;successful=false) = new(successful)
end
@declare_hyperparameters(FitParamT, [:successful])

function fit!(self::FitParamT, X, y; should_succeed=false)
    self.successful = should_succeed
end

function predict(self::FitParamT, X)
    return self.successful
end

############################################################

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
    # (Note: I think this should be an ArgumentError if this was a Julia
    # estimator)
    @test_throws(PyError, set_params!(pipe; anova__C=0.1))

    # Test clone
    pipe2 = clone(pipe)
    @test named_steps(pipe)["svc"] !== named_steps(pipe2)["svc"]

    # Check that apart from estimators, the parameters are the same
    params = get_params(pipe)
    params2 = get_params(pipe2)
    # Remove estimators that where copied
    delete!(params, "svc")
    delete!(params, "anova")
    delete!(params2, "svc")
    delete!(params2, "anova")
    @test params == params2
end


function test_pipeline_methods_anova()
    # Test the various methods of the pipeline (anova).
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    # Test with Anova + LogisticRegression
    clf = LogisticRegression()
    filter1 = SelectKBest(f_classif, k=2)
    pipe = Pipeline([("anova", filter1), ("logistic", clf)])
    fit!(pipe, X, y)
    predict(pipe, X)
    predict_proba(pipe, X)
    predict_log_proba(pipe, X)
    score(pipe, X, y)
end


function test_pipeline_fit_params()
    # Test that the pipeline can take fit parameters
    pipe = Pipeline([("transf", T()), ("clf", FitParamT())])
    # Julia note: fit! does not currently support arbitrary keyword arguments.
    # This feature doesn't seem to be used so far, and it looks like an
    # historical artifact to me since `set_params!` is used to achieve the
    # same thing - cstjean Feb2016
    # fit!(pipe, 1, 2; clf__should_succeed=true)
    # Replacement test
    set_params!(pipe; clf__successful=true)
    # classifier should return true
    @test predict(pipe, nothing)
    # and transformer params should not be changed
    @test named_steps(pipe)["transf"].a == nothing
    @test named_steps(pipe)["transf"].b == nothing
end


function test_pipeline_methods_pca_svm()
    # Test the various methods of the pipeline (pca + svm).
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    # Test with PCA + SVC
    clf = SVC(probability=true, random_state=0)
    pca = PCA(n_components="mle", whiten=true)
    pipe = Pipeline([("pca", pca), ("svc", clf)])
    fit!(pipe, X, y)
    predict(pipe, X)
    predict_proba(pipe, X)
    predict_log_proba(pipe, X)
    score(pipe, X, y)
end


function test_pipeline_methods_preprocessing_svm()
    # Test the various methods of the pipeline (preprocessing + svm).
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    n_samples = size(X, 1)
    n_classes = length(unique(y))
    scaler = StandardScaler()
    pca = RandomizedPCA(n_components=2, whiten=true)
    clf = SVC(probability=true, random_state=0)

    for preprocessing in [scaler, pca]
        pipe = Pipeline([("preprocess", preprocessing), ("svc", clf)])
        fit!(pipe, X, y)

        # check shapes of various prediction functions
        pred = predict(pipe, X)
        @test size(pred) == (n_samples,)

        proba = predict_proba(pipe, X)
        @test size(proba) == (n_samples, n_classes)

        log_proba = predict_log_proba(pipe, X)
        @test size(log_proba) == (n_samples, n_classes)

        dec_function = decision_function(pipe, X)
        @test size(dec_function) == (n_samples, n_classes)

        score(pipe, X, y)
    end
end


function test_feature_union()
    # basic sanity check for feature union
    iris = load_iris()
    X = iris["data"]
    X = X .- mean(X, 1)
    y = iris["target"]
    svd = TruncatedSVD(n_components=2, random_state=0)
    select = SelectKBest(k=1)
    fs = FeatureUnion([("svd", svd), ("select", select)])
    fit!(fs, X, y)
    X_transformed = transform(fs, X)
    @test size(X_transformed) == (size(X, 1), 3)

    # check if it does the expected thing
    @test isapprox(X_transformed[:, 1:end-1], fit_transform!(svd, X))
    @test(X_transformed[:, end] ==
          fit_transform!(select, X, y)[:])

    # test if it also works for sparse input
    # We use a different svd object to control the random_state stream
    # TODO
    ## fs = FeatureUnion([("svd", svd), ("select", select)])
    ## X_sp = sparse.csr_matrix(X)
    ## X_sp_transformed = fs.fit_transform(X_sp, y)
    ## assert_array_almost_equal(X_transformed, X_sp_transformed.toarray())

    # test setting parameters
    set_params!(fs; select__k=2)
    @test size(fit_transform!(fs, X, y)) == (size(X, 1), 4)

    # test it works with transformers missing fit_transform
    fs = FeatureUnion([("mock", T()), ("svd", svd), ("select", select)])
    X_transformed = fit_transform!(fs, X, y)
    @test size(X_transformed) == (size(X, 1), 8)
end


function test_pipeline_transform()
    # Test whether pipeline works with a transformer at the end.
    # Also test pipeline.transform and pipeline.inverse_transform
    iris = load_iris()
    X = iris["data"]
    pca = PCA(n_components=2)
    pipeline = Pipeline([("pca", pca)])

    # test transform and fit_transform:
    X_trans = transform(fit!(pipeline, X), X)
    X_trans2 = fit_transform!(pipeline, X)
    X_trans3 = fit_transform!(pca, X)
    @test isapprox(X_trans, X_trans2)
    @test isapprox(X_trans, X_trans3)

    X_back = inverse_transform(pipeline, X_trans)
    X_back2 = inverse_transform(pca, X_trans)
    @test isapprox(X_back, X_back2)
end


function test_pipeline_fit_transform()
    # Test whether pipeline works with a transformer missing fit_transform
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    transft = T()
    pipeline = Pipeline([("mock", transft)])

    # test fit_transform:
    X_trans = fit_transform!(pipeline, X, y)
    X_trans2 = transform(fit!(transft, X, y), X)
    @test isapprox(X_trans, X_trans2)
end


function test_make_pipeline()
    t1 = T()
    t2 = T()

    pipe = make_pipeline(t1, t2)
    @test(isa(pipe, Pipeline))
    @test(pipe.steps[1][1]=="t-1")
    @test(pipe.steps[2][1]=="t-2")

    pipe = make_pipeline(t1, t2, FitParamT())
    @test(isa(pipe, Pipeline))
    @test(pipe.steps[1][1]=="t-1")
    @test(pipe.steps[2][1]=="t-2")
    @test(pipe.steps[3][1]=="fitparamt")
end


function test_feature_union_weights()
    # test feature union with transformer weights
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    pca = RandomizedPCA(n_components=2, random_state=0)
    select = SelectKBest(k=1)
    # test using fit followed by transform
    fs = FeatureUnion([("pca", pca), ("select", select)],
                      transformer_weights=Dict("pca"=>10))
    fit!(fs, X, y)
    X_transformed = transform(fs, X)
    # test using fit_transform
    fs = FeatureUnion([("pca", pca), ("select", select)],
                      transformer_weights=Dict("pca"=>10))
    X_fit_transformed = fit_transform!(fs, X, y)
    # test it works with transformers missing fit_transform
    fs = FeatureUnion([("mock", T()), ("pca", pca), ("select", select)],
                      transformer_weights=Dict("mock"=>10))
    X_fit_transformed_wo_method = fit_transform!(fs, X, y)
    # check against expected result

    # We use a different pca object to control the random_state stream
    @test isapprox(X_transformed[:, 1:end-1], 10 * fit_transform!(pca, X))
    @test X_transformed[:, end] == fit_transform!(select, X, y)[:]
    @test isapprox(X_fit_transformed[:, 1:end-1], 10 * fit_transform!(pca, X))
    @test X_fit_transformed[:, end] == fit_transform!(select, X, y)[:]
    @test size(X_fit_transformed_wo_method) == (size(X, 1), 7)
end


# TODO
# def test_feature_union_parallel():
# def test_feature_union_feature_names():

function test_classes_property()
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]

    reg = make_pipeline(SelectKBest(k=1), LinearRegression())
    fit!(reg, X, y)
    @test_throws(KeyError, get_classes(reg))

    clf = make_pipeline(SelectKBest(k=1), LogisticRegression(random_state=0))
    @test_throws(KeyError, get_classes(clf))
    fit!(clf, X, y)
    @test get_classes(clf) == unique(y)
end


function all_test_pipeline()
    test_pipeline_init()
    test_pipeline_methods_anova()
    test_pipeline_fit_params()
    test_pipeline_methods_pca_svm()
    test_pipeline_methods_preprocessing_svm()
    test_feature_union()
    test_pipeline_transform()
    test_pipeline_fit_transform()
    test_make_pipeline()
    test_feature_union_weights()
    test_classes_property()
end
