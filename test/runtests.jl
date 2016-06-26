println("Testing MinFinder started.")
using MinFinder
using Base.Test

@test 1 == 1

ex = MinFinder.MultipleMinimaProblems.examples

function test_multiple(problem::MinFinder.MultipleMinimaProblems.OptimizationProblem,
    seed::Int=1; kwargs...)
    @printf "%s, seed=%s \n" problem.name seed

    srand(seed)
    res = Optim.optimize(problem.f, problem.l, problem.u, Fminfinder(); kwargs...)
    mins = Optim.minimizer(res)
    @test length(mins) == length(problem.minima)
    #@assert length(mins)==length(problem.minima)
    for m in mins
        foundmin = false
        for i in 1:length(problem.minima)
            if norm(m - problem.minima[i], 2) < 1e-4 * length(problem.l)
                foundmin = true
            end
        end
        #foundmin || println(name, m)
        @assert foundmin
    end
end

test_multiple(ex["Rosenbrock"])
test_multiple(ex["Camel"])
#test_multiple(ex["Rastrigin"], 6)
test_multiple(ex["Shekel5"])
test_multiple(ex["Shekel7"])
test_multiple(ex["Shekel10"])

println("minfinder test successful")
