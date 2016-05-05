using Base.Test
using Skcore: type_of_target

@test type_of_target([0.1, 0.6]) == "continuous"
@test type_of_target([1, -1, -1, 1]) == "binary"
@test type_of_target(["a", "b", "a"]) == "binary"
@test type_of_target([1.0, 2.0]) == "continuous"
@test type_of_target([1, 0, 2]) == "multiclass"
@test type_of_target([1.0, 0.0, 3.0]) == "continuous"
@test type_of_target(["a", "b", "c"]) == "multiclass"
@test type_of_target([1 2; 3 1]) == "multiclass-multioutput"
@test type_of_target([1.5 2.0; 3.0 1.6]) == "continuous-multioutput"
@test type_of_target([0 1; 1 1]) == "multilabel-indicator"
