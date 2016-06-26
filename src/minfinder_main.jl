# TODO: is the check that a new minima was already found correct (norm<tol)?
# TODO: the 2008 paper introduces non-gradient based checking rules. Thus, a
#        derivative-free MinFinder could be implement that also uses derivative-
#        free local searches
# TODO: add parallel computing for launching searches
# TODO: add more tests

"Type for both the starting point and resulting minima of a local search."
immutable SearchPoint{T}
   x::Vector{T} # point in parameter space
   g::Vector{T} # gradient at point `x` (can be `nothing`)
   val::T       # function value at point `x`
end
SearchPoint{T}(x::Vector{T}, g::Vector{T}) = SearchPoint(x, g, convert(T, NaN))
SearchPoint{T}(x::Vector{T}) = SearchPoint(x, Vector{T}())

# Check starting point `a` against SearchPoint `b` that could be either another
# starting point or found minima.
function isvalidpoint{T}(a::SearchPoint{T}, b::SearchPoint{T}, dist)
    #StatsBase.L2dist(a.x, b.x) > dist && dot(a.x - b.x, a.g - b.g) < 0
    s = zero(T)
    t = zero(T)
    for i in eachindex(a.x)
        dx = a.x[i] - b.x[i]
        s += dx * dx
        t += dx * (a.g[i] - b.g[i])
    end
    return sqrt(s) > dist && t < 0
end
function isvalidpoint{T}(p::SearchPoint{T}, v::Vector{SearchPoint{T}}, dist)
    valid = true
    for q in v
        if !isvalidpoint(p, q, dist)
            valid = false
            break
        end
    end
    valid
end

"""
Implementation based on the papers (not on the accompanying code):
* Ioannis G. Tsoulos, Isaac E. Lagaris, MinFinder: Locating all the local minima
of a function, Computer Physics Communications, Volume 174, January 2006,
Pages 166-179. http://dx.doi.org/10.1016/j.cpc.2005.10.001
* Ioannis G. Tsoulos, Isaac E. Lagaris, MinFinder v2.0: An improved version of
MinFinder, Computer Physics Communications, Volume 179, Issue 8,
15 October 2008, Pages 614-615, ISSN 0010-4655
http://dx.doi.org/10.1016/j.cpc.2008.04.016.

From the abstract: "A new stochastic clustering algorithm is introduced that
aims to locate all the local minima of a multidimensional continuous and
differentiable function inside a bounded domain. [..] We compare the
performance of this new method to the performance of Multistart and
Topographical Multilevel Single Linkage Clustering on a set of benchmark
problems."

Because the search domain is bounded, minfinder uses `Fminbox` for the
individual optimizations that uses `ConjugateGradient` as local optimizer
by default.

Algortihm parameters as in paper:
Nmax = "predefined upper limit for the number of samples in each
       generation. This  step prevents the algorithm from performing an
       insufficient exploration of the search space."
Ninit = initial number of samples
exhaustive =  "[...] in the range (0,1). For small values of p (p→0) the
        algorithm searches the area exhaustively, while for p→1, the
        algorithm terminates earlier, but perhaps prematurely."

When using polish, stopping tolerances are set quite high by default
(sqrt of usual tol). At the end the minima are polished off. Inspiration
from S. Johnson: [http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms#MLSL_.28Multi-Level_Single-Linkage.29]

Other options:
polish: Perform final optimization on each found minima?
local_tol: tolerance level for local searches
polish_tol: tolerance level for final polish of minima
distmin: discard minima is closer than distmin to found minima
distpolish: same for final minima polish
max_algo_steps: maximum number of minfinder steps (each with N points sampled)
show_trace: show progress
"""
function Optim.optimize{T <: AbstractFloat}(
    df::Optim.DifferentiableFunction,
    l::Array{T},
    u::Array{T},
    ::Fminfinder;
    enrich = 1.1,
    Nmax::Integer = 250,
    Ninit::Integer = 20,
    exhaustive = 0.5,
    max_algo_steps::Integer = 1_000,
    show_trace::Bool = false,
    polish::Bool = true,
    local_xtol = (polish ? sqrt(eps(T)) : eps(T)),
    local_ftol = (polish ? sqrt(eps(T)^(2/3)) : eps(T)^(2/3)),
    local_grtol = (polish ? sqrt(eps(T)^(2/3)) : eps(T)^(2/3)),
    #method::Optim.Optimizer = Optim.ConjugateGradient,
    polish_xtol = eps(T),
    polish_ftol = eps(T)^(2/3),
    polish_grtol = eps(T)^(2/3),
    distmin = sqrt(local_xtol),
    distpolish = sqrt(polish_xtol))

    @assert length(l) == length(u)

    # Initiate
    N = Ninit # number of starting point samples
    typical_distance = zero(T) # typical distance between start and its minima
    min_distance = convert(T, Inf) #for use in ValidPoint: min distance between minima
    stoplevel = 0.0 # 'a' in paper = exhaustive * var_last

    minima = Vector{SearchPoint{T}}() # found minima
    polishminina = similar(minima) # mimina after final polish
    iterminima   = similar(minima) #minima found during one MinFinder iteration
    startpoints  = similar(minima) #starting points for local minimizations

    x   = similar(l) # temporary function point input
    val = zero(T) # temporary function value
    g   = similar(l) # temporary function gradient at point
    p   = SearchPoint(x, g, val) #temporary SearchPoint

    all_f_calls = 0 #number of total function evaluations
    all_g_calls = 0 #number of total gradient evaluations
    algo_steps  = 0 #number of minfinder iterations
    searches    = 0 #number of local minimizations
    converges   = 0 #number of converged searches

    # Define stopping rule of the paper. In short, create a series of binomial
    # events from 1 to N. The variance of this series goes slowly to zero.
    # Compare this value with `stoplevel` at the latest iteration when a minima
    # was found.
    # doublebox(n::Int) = var([StatsBase.rand_binom(i, .5)/i for i=1:n])
    # StatsBase.rand_binom does not work with julia v0.2.1, so sum bernoulli
    doublebox(n::Int) = var([sum(round(Int, rand(i)))/i for i = 1:n])

    if show_trace
        @printf "############### minfinder ############### \n"
        @printf "Steps  N    Searches   Function Calls   Minima \n"
        @printf "-----  ---  --------   --------------   ------ \n"
    end


    function sample_check!(startpoints) # Sampling and checking step
        empty!(startpoints)
        dim = length(l)
        for _ = 1:N
            x = l + rand(dim) .* (u - l)
            df.g!(x, g) # no function value required for checking rule
            all_g_calls += 1
            p = SearchPoint(x, copy(g))

            # check each point before accepting as starting point
            if !isempty(minima) # Note there's no typical_distance without minima
                # condition 1: check against all other starting points
                isvalidpoint(p, startpoints, typical_distance) || continue
                # condition 2: check against previously found minima
                isvalidpoint(p, minima, min_distance) || continue
            end
            push!(startpoints, p)
        end
    end

    # main MinFinder algorithm loop
    while (doublebox(N) > stoplevel) & (algo_steps < max_algo_steps)
        algo_steps += 1

        sample_check!(startpoints)
        empty!(iterminima)

        # Enrichment for next iteration
        length(startpoints) < N/2 && (N = min(round(Int, N * enrich), Nmax))

        for p in startpoints
            # Check start point again in case minima found at current iteration
            isvalidpoint(p, iterminima, min_distance) || continue

            result = optimize(df, p.x, l, u, Fminbox();#, method;
                        xtol=local_xtol, ftol=local_ftol, rtol=local_grtol)
            searches    += 1
            all_f_calls += Optim.f_calls(result)
            all_g_calls += Optim.g_calls(result)
            x            = Optim.minimizer(result)
            val          = Optim.minimum(result)
            hasconverged = Optim.converged(result)

            if hasconverged
                converges += 1

                # Update typical search distance (by streaming average)
                typical_distance = (typical_distance * (searches - 1) +
                    norm(p.x - x, 2)) / searches

                # add to minima if not found earlier
                minfound = false
                for m in minima
                    if norm(x - m.x, 2) < distmin
                        minfound = true
                        break
                    end
                end
                minfound && continue

                df.g!(x, g) # gradient not given with optimize result
                all_g_calls += 1
                push!(iterminima, SearchPoint(x, copy(g), val))
                push!(minima,     SearchPoint(x, copy(g), val))

                min_distance = min(min_distance, norm(p.x - x, 2))

                # Update stoplevel TODO is this at correct place?
                stoplevel = exhaustive * doublebox(N)

            end #if converged
        end #for startpoints

        if show_trace
            @printf "%4d   %3d   %8d   %14d   %6d\n" algo_steps N searches all_f_calls length(minima)
        end
    end #while


    # Polish off minima
    if polish
        for m in minima
            result = optimize(df, m.x, l, u, Fminbox();#, method;
                        xtol=polish_xtol, ftol=polish_ftol, grtol=polish_grtol)
            searches    += 1
            all_f_calls += Optim.f_calls(result)
            all_g_calls += Optim.g_calls(result)
            x            = Optim.minimizer(result)
            val          = Optim.minimum(result)
            hasconverged = Optim.converged(result)

            # Check if not converged to another final optimization minima
            minfound = false
            for h in polishminina
                if norm(x - h.x,2) < distpolish
                    minfound = true
                    break
                end
            end
            if !minfound
                df.g!(x, g)
                all_g_calls += 1
                push!(polishminina, SearchPoint(x, copy(g), val))
            end
        end

        if show_trace
            @printf "Final polish retained %d minima out of %d \n" length(polishminina) length(minima)
        end
    end #if polish

    finalminima = ifelse(polish, polishminina, minima)

    return FminfinderOptimizationResults(
        l,
        u,
        map(m -> m.x, finalminima),
        map(m -> m.g, finalminima),
        all_f_calls,
        all_g_calls,
        searches,
        algo_steps,
        N,
        N == Nmax,
        minima,
        typical_distance,
        min_distance,
        converges,
        polish)

end #funtion

type FminfinderOptimizationResults{T} <: Optim.OptimizationResults
    initial_lower :: Vector{T}
    initial_upper :: Vector{T}
    minima        :: Vector{Vector{T}}
    f_minima      :: Vector{Vector{T}}
    f_calls       :: Int
    g_calls       :: Int
    searches      :: Int
    algo_steps    :: Int
    N_last        :: Int
    N_equals_Nmax :: Bool
    minima_unpolished::Vector{SearchPoint{T}}
    typical_distance ::T
    min_distance  :: T
    converges     :: Int
    polished      :: Bool
end

function Base.show(io::IO, r::FminfinderOptimizationResults)
    println(io, "Fminbox optimizatio result with $(length(Optim.minimum(r))) minima.")
end

 Optim.iterations(r::FminfinderOptimizationResults) = r.N
  Optim.minimizer(r::FminfinderOptimizationResults) = r.minima
    Optim.minimum(r::FminfinderOptimizationResults) = r.f_minima
  Optim.converged(r::FminfinderOptimizationResults) = r.converges > 0
Optim.lower_bound(r::FminfinderOptimizationResults) = r.initial_lower
Optim.upper_bound(r::FminfinderOptimizationResults) = r.initial_upper
     Optim.method(r::FminfinderOptimizationResults) = "Fminfinder"
iteration_limit_reached(r::FminfinderOptimizationResults) = r.N_equals_Nmax
