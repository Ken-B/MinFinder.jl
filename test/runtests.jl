println("Testing MinFinder started.")
using MinFinder
using Base.Test


ex = MinFinder.MultipleMinimaProblems.examples

function test_multiple(problem::MinFinder.MultipleMinimaProblems.OptimizationProblem;
    seed::Int=1, kwargs...)
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
test_multiple(ex["Rastrigin"], seed=1)
test_multiple(ex["Shekel5"])
test_multiple(ex["Shekel7"])
test_multiple(ex["Shekel10"], seed=2)

println("Rosenbrock Float32")
f32min = Optim.minimum(optimize(ex["Rosenbrock"].f, [-5f0, -5f0], [5f0, 5f0], Fminfinder()))
@test eltype(f32min) == Float32
@test_approx_eq_eps [0f0] f32min 1e-8
println("minfinder test successful")
