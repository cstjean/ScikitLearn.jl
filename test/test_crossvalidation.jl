using Base.Test
using Skcore
using Skcore: cross_val_score
using PyCall: PyError
using SklearnBase: @simple_estimator_constructor

@pyimport2 sklearn.cross_validation as cval


## function check_valid_split(train, test, n_samples=None)
##     # Use python sets to get more informative assertion failure messages
##     train, test = set(train), set(test)

##     # Train and test split should not overlap
##     assert_equal(train.intersection(test), set())

##     if n_samples is not None:
##         # Check that the union of train an test split cover all the indices
##         assert_equal(train.union(test), set(range(n_samples)))
##     end
## end


"""Dummy classifier to test the cross-validation"""
type MockClassifier
    a
    allow_nd
    dummy_int
    dummy_str
    dummy_obj
end
@simple_estimator_constructor MockClassifier(; a=0, allow_nd=false) =
    MockClassifier(a, allow_nd, nothing, nothing, nothing)
Skcore.is_classifier(::MockClassifier) = true


"""The dummy arguments are to test that this fit function can
accept non-array arguments through cross-validation, such as:
    - int
    - str (this is actually array-like)
    - object
    - function
"""
function Skcore.fit!(self::MockClassifier, X, Y=nothing; sample_weight=nothing,
                     class_prior=nothing,
                     sparse_sample_weight=nothing, sparse_param=nothing,
                     dummy_int=nothing,
                     dummy_str=nothing, dummy_obj=nothing, callback=nothing)
    self.dummy_int = dummy_int
    self.dummy_str = dummy_str
    self.dummy_obj = dummy_obj
    if callback != nothing
        callback(self)
    end

    if self.allow_nd
        X = reshape(X, size(X, 1), Int(prod(size(X)) / size(X, 1)))
    end
    if ndims(X) >= 3 && !self.allow_nd
        throw(ArgumentError("X cannot be d"))
    end
    if sample_weight !== nothing
        ## 'MockClassifier extra fit_param sample_weight.shape[0]'
        ## ' is {0}, should be {1}'.format(sample_weight.shape[0],
        ##                                 X.shape[0]))
        @test size(sample_weight, 1) == size(X, 1)
    end
    if class_prior !== nothing
        ## 'MockClassifier extra fit_param class_prior.shape[0]'
        ## ' is {0}, should be {1}'.format(class_prior.shape[0],
        ##                                 len(np.unique(y))))
        @test size(class_prior, 1) == length(unique(y))
    end
    if sparse_sample_weight !== nothing
        ## fmt = ('MockClassifier extra fit_param sparse_sample_weight'
        ##        '.shape[0] is {0}, should be {1}')
        @test size(sparse_sample_weight, 1) == size(X, 1)
    end
    if sparse_param !== nothing
        ## fmt = ('MockClassifier extra fit_param sparse_param.shape '
        ##        'is ({0}, {1}), should be ({2}, {3})')
        @test size(sparse_param) == size(P_sparse)
    end
    return self
end


function Skcore.predict(self::MockClassifier, T)
    if self.allow_nd
        T = reshape(T, size(T, 1), Int(length(T) / size(T, 1)))
    end
    return size(T, 1)
end

Skcore.score(self::MockClassifier, X=nothing, Y=nothing) =
    1. / (1 + abs(self.a))


X = ones(10, 2)
X_sparse = sparse(X)
## W_sparse = coo_matrix((np.array([1]), (np.array([1]), np.array([0]))),
##                       shape=(10, 1))
P_sparse = sparse(eye(5))
y = [floor(Int, x / 2) for x in 0:9]

################################################################################
# Tests

# There are a lot of tests in the Python versions that are just for generating
# good folds, but since we use Python, we don't really need to test them.

function test_cross_val_score()
    clf = MockClassifier()
    for a in -10:9
        clf.a = a
        # Smoke test
        scores = cross_val_score(clf, X, y)
        @test all(scores .== score(clf, X, y))

        # test with multioutput y - Julia TODO
        ## scores = cross_val_score(clf, X_sparse, X)
        ## assert_array_equal(scores, clf.score(X_sparse, X))

        ## scores = cross_val_score(clf, X_sparse, y)
        ## assert_array_equal(scores, clf.score(X_sparse, y))

        # test with multioutput y
        ## scores = cval.cross_val_score(clf, X_sparse, X)
        ## assert_array_equal(scores, clf.score(X_sparse, X))
    end

    # test with X and y as list - Julia TODO
    # CheckingClassifier comes from sklearn.utils.mocking
    ## list_check = lambda x: isinstance(x, list)
    ## clf = CheckingClassifier(check_X=list_check)
    ## scores = cval.cross_val_score(clf, X.tolist(), y.tolist())

    ## clf = CheckingClassifier(check_y=list_check)
    ## scores = cval.cross_val_score(clf, X, y.tolist())

    ## assert_raises(ValueError, cval.cross_val_score, clf, X, y,
    ##               scoring="sklearn")

    # test with 3d X
    X_3d = reshape(X, size(X)..., 1)
    ## clf = MockClassifier(allow_nd=true)
    ## scores = cross_val_score(clf, X_3d, y)

    ## clf = MockClassifier(allow_nd=false)
    ## @test_throws(ArgumentError, cross_val_score(clf, X_3d, y))
end

# def test_cross_val_score_pandas(): TODO Julia

# This test was for deprecated sklearn-py behaviour (CV iterators returning
# masks instead of indices), so it doesn't matter anymore.
## function test_cross_val_score_mask()
##     # test that cross_val_score works with boolean masks
##     svm = SVC(kernel="linear")
##     iris = load_iris()
##     X, y = iris["data"], iris["target"]
##     cv_indices = cval.KFold(length(y), 5, indices=true)
##     scores_indices = cross_val_score(svm, X, y, cv=cv_indices)
##     cv_masks = cval.KFold(length(y), 5, indices=false)
##     scores_masks = cross_val_score(svm, X, y, cv=cv_masks)
##     @test scores_indices == scores_masks
## end

# I think this test is related to the checks at the top of _safe_split. TODO
#def test_cross_val_score_precomputed()


function test_cross_val_score_fit_params()
    clf = MockClassifier()
    n_samples = size(X, 1)
    n_classes = length(unique(y))

    DUMMY_INT = 42
    DUMMY_STR = "42"
    DUMMY_OBJ = Dict()

    function assert_fit_params(clf)
        # Function to test that the values are passed correctly to the
        # classifier arguments for non-array type

        @test clf.dummy_int == DUMMY_INT
        @test clf.dummy_str == DUMMY_STR
        @test clf.dummy_obj == DUMMY_OBJ
    end
    
    fit_params = Dict(:sample_weight=> ones(n_samples),
                      :class_prior=> ones(n_classes) / n_classes,
                      #:sparse_sample_weight=> W_sparse, TODO
                      :sparse_param=> P_sparse,
                      :dummy_int=> DUMMY_INT,
                      :dummy_str=> DUMMY_STR,
                      :dummy_obj=> DUMMY_OBJ,
                      :callback=> assert_fit_params)
    cross_val_score(clf, X, y, fit_params=fit_params)
end


function test_cross_val_score_score_func()
    clf = MockClassifier()
    _score_func_args = []

    function score_func(y_test, y_predict)
        push!(_score_func_args, (y_test, y_predict))
        return 1.0
    end

    scoring = make_scorer(score_func)
    score = cross_val_score(clf, X, y, scoring=scoring)
    @test score == [1.0, 1.0, 1.0]
    @test length(_score_func_args) == 3
end


function all_test_crossvalidation()
    test_cross_val_score()
    test_cross_val_score_fit_params()
    # Disabled until we have `make_scorer`
    #test_cross_val_score_score_func()
end
