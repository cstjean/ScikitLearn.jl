module TestingLinearRegression

using Test
using ScikitLearn.Models: LinearRegression, fit!, predict

close_enough(x0, x1, tol=1e-14) = abs(x0 - x1) <= tol ? true : false

@testset "Single-Target Keyword Tests" begin

    @testset "One-independent variable Tests" begin
        # Matrices should be n_features x n_samples
        X = [1 2 3 4 5]
        y = identity.(X)

        @test begin
            y_pred = predict(fit!(LinearRegression(), X, y), X)
            all(map(close_enough, y_pred, y))
        end
        @test begin
            y_pred = predict(fit!(LinearRegression(multi_output=false), X, y), X)
            all(map(close_enough, y_pred, y))
        end
    end

    @testset "Two-independent variables Tests" begin
         X = [1 2 3 4 5 ;
              2 4 6 8 10]
         y = mapslices(x -> x[1] - x[2], X, dims=1)

         @test begin
             y_pred = predict(fit!(LinearRegression(), X, y), X)
             all(map(close_enough, y_pred, y))
         end
         @test begin
             y_pred = predict(fit!(LinearRegression(multi_output=false), X, y), X)
             all(map(close_enough, y_pred, y))
         end
    end
end

@testset "Multi-Target Keyword Tests" begin
    # Matrices should be n_features x n_samples
    @testset "One-independent variable Tests" begin
        X = [1 2 3 4 5]
        y = [1  2  3  4  5;
            -1 -2 -3 -4 -5]
        @test begin
            y_pred = predict(fit!(LinearRegression(), X, y), X)
            all(map(close_enough, y_pred, y))
        end
        @test begin
            y_pred = predict(fit!(LinearRegression(multi_output=true), X, y), X)
            all(map(close_enough, y_pred, y))
        end
    end

    @testset "Two-independent variable Tests" begin
        X = [1 2 3 4 5 ;
             2 4 6 8 10]
        y = mapslices(x -> [x[1] + x[2], x[1] - x[2]], X, dims=1)

        @test begin
            y_pred = predict(fit!(LinearRegression(), X, y), X)
            all(map(close_enough, y_pred, y))
        end
        @test begin
            y_pred = predict(fit!(LinearRegression(multi_output=true), X, y), X)
            all(map(close_enough, y_pred, y))
        end
    end
end

end # module
