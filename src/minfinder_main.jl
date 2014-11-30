### MinFinder ###
# Syntax:
#    `minima, f_calls, g_calls, searches, steps = minfinder(df, l, u)`
# Inputs:
#   `df` is of type DifferentiableFunction as from `Optim` package
#   `l` contain the lower boundaries of the search domain
#   `u` contain the upper boundaries of the search domain
# Outputs:
#    minima is a Vector that contains SearchPoint types
#    f_calls is the number of function evaluations
#    g_calls is the number of gradient evaluations
#    searches is the number of local minimizations performed
#    steps is the number of minfinder steps before stopping rule hit
# 
# Based on the papers:
# Ioannis G. Tsoulos, Isaac E. Lagaris, MinFinder: Locating all the local minima
# of a function, Computer Physics Communications, Volume 174, January 2006, 
# Pages 166-179. http://dx.doi.org/10.1016/j.cpc.2005.10.001
#
# Ioannis G. Tsoulos, Isaac E. Lagaris, MinFinder v2.0: An improved version of 
# MinFinder, Computer Physics Communications, Volume 179, Issue 8, 
# 15 October 2008, Pages 614-615, ISSN 0010-4655
# http://dx.doi.org/10.1016/j.cpc.2008.04.016.
#
# From the abstract: "A new stochastic clustering algorithm is introduced that 
# aims to locate all the local minima of a multidimensional continuous and 
# differentiable function inside a bounded domain. [..] We compare the 
# performance of this new method to the performance of Multistart and 
# Topographical Multilevel Single Linkage Clustering on a set of benchmark 
# problems."
#
# Because the search domain is bounded, minfinder uses `fminbox` for local 
# searches from the `Optim` pacakge using `cg` by default.
#
# TODO: is the check that a new minima was already found correct (norm<tol)?
# TODO: the 2008 paper introduces non-gradient based checking rules. Thus a
#        derivative-free MinFinder could be implement that also uses derivative-
#        free local searches
# TODO: add parallel computing

# Create types for the starting points and the minima
type SearchPoint{T} 
    x::Vector{T} # point
    g::Vector{T} # gradient at point (can be nothing)
    val::T       # function value at point    
end
SearchPoint{T}(x::Vector{T}, g::Vector{T}) = SearchPoint(x, g, nan(T))
SearchPoint{T}(x::Vector{T}) = SearchPoint(x, Array(T,0))

function minfinder{T <: FloatingPoint}(df::DifferentiableFunction, 
    l::Array{T,1}, 
    u::Array{T,1};
    enrich = 1.1,
    Nmax::Integer = 250,
    Ninit::Integer = 20,
    exhaustive = .5,
    max_algo_steps::Integer = 1_000,
    show_trace::Bool = false,
    polish::Bool = true, 
    local_xtol = (polish ? sqrt(eps(T)) : eps(T)), 
    local_ftol = (polish ? sqrt(eps(T)^(2/3)) : eps(T)^(2/3)),
    local_grtol = (polish ? sqrt(eps(T)^(2/3)) : eps(T)^(2/3)),
    method = :cg,
    polish_xtol = eps(T),
    polish_ftol = eps(T)^(2/3),
    polish_grtol = eps(T)^(2/3),
    distmin = sqrt(local_xtol),
    distpolish = sqrt(polish_xtol))

    # Algortihm parameters as in paper:
    # Nmax = "predefined upper limit for the number of samples in each 
    #        generation. This  step prevents the algorithm from performing an 
    #        insufficient exploration of the search space."
    # Ninit = initial number of samples 
    # exhaustive =  "..in the range (0,1). For small values of p (p→0) the 
    #         algorithm searches the area exhaustively, while for p→1, the 
    #         algorithm terminates earlier, but perhaps prematurely."

    # When using polish, stopping tolerances are set quite high by default 
    # (sqrt of usual tol). At the end the minima are polished off. Inspiration 
    # from S. Johnson: [http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms#MLSL_.28Multi-Level_Single-Linkage.29]

    # Other options:
    # polish: Perform final optimization on each found minima?
    # local_tol: tolerance level for local searches
    # polish_tol: tolerance level for final polish of minima    
    # distmin: discard minima is closer than distmin to found minima
    # distpolish: same for final minima polish
    # max_algo_steps: maximum number of minfinder steps (each with N points sampled)
    # show_trace: show progress

    length(l) == length(u) ||error("boundary vectors must have the same length")

    # Initiate
    N = Ninit # number of starting point samples 
    typical_distance = zero(T) # typical distance between start and its minima
    min_distance = inf(T) #for use in ValidPoint: min distance between minima    
    stoplevel = 0. # 'a' in paper = exhaustive * var_last

    minima = Array(SearchPoint{T}, 0) # type with found minima
    polishminina = Array(SearchPoint{T}, 0) # mimina after final polish
    iterminima = Array(SearchPoint{T}, 0) #minima found during one iteration
    points = Array(SearchPoint{T}, 0) #starting points for local minimizations

    x = similar(l) # temporary function point input
    val = zero(T) # temporary function value
    g = similar(l) # temporary function gradient at point
    p = SearchPoint(x, g, val) #temporary SearchPoint

    f_calls::Int = 0 #number of function evaluations
    g_calls::Int = 0 #number of gradient evaluations
    algo_steps::Int = 0 #number of minfinder iterations
    searches::Int = 0 #number of local minimizations
    converges::Int = 0 #number of converged searches

    # Define stopping rule of the paper. In short, create a series of binomial
    # events from 1 to N. The variance of this series goes slowly to zero. 
    # Compare this value with `stoplevel` at the latest iteration when a minima 
    # was found. 
    # doublebox(n::Int) = var([StatsBase.rand_binom(i, .5)/i for i=1:n])
    # StatsBase.rand_binom does not work with julia v0.2.1, so sum bernoulli
    doublebox(n::Int) = var([sum(int(rand(i)))/i for i=1:n])

    dim = length(l) #precalc dimension of problem
    function checkrule{T}(a::SearchPoint{T}, b::SearchPoint{T}, dist)
        #L2dist(a.x, b.x) < dist && dot(a.x - b.x, a.g - b.g) > 0
        ax = a.x
        bx = b.x
        ag = a.g
        bg = b.g
        s = zero(T)
        t = zero(T)
        for i = 1:dim
            dx = ax[i] - bx[i]
            s += dx * dx
            t += dx * (ag[i] - bg[i])
        end
        return sqrt(s) < dist && t > 0
    end

    # Show progress
    if show_trace
        @printf "############### minfinder ############### \n"
        @printf "Steps  N    Searches   Function Calls   Minima \n"
        @printf "-----  ---  --------   --------------   ------ \n"
    end
    
    # main minfinder algorithm loop
    while (doublebox(N) > stoplevel) & (algo_steps < max_algo_steps)
        algo_steps += 1

        # Sampling and checking step
        points = Array(SearchPoint{T}, 0) #empty points array
        for unused=1:N
            x = l + rand(dim) .* (u - l)
            df.g!(x, g) # no function value required for checkrule
            g_calls += 1
            p = SearchPoint(x, copy(g))

            # check on each point before accepting as starting point
            validpoint = true
            if !isempty(minima) # no typical_distance without minima        
                # condition 1: check against all other points in `pnts`
                for q in points
                    if checkrule(p, q, typical_distance); validpoint=false; end
                end
                # condition 2: check against found minima in `mins`
                for z in minima
                    if checkrule(p, z, min_distance);validpoint=false; end
                end
            end 
            validpoint && push!(points, p)
        end

        # Enrichment for next iteration
        if length(points) < N/2 
            N = min(int(N * enrich), Nmax)
        end

        iterminima = Array(SearchPoint{T}, 0) #clear iterminima
        for p in points

            # If minima found during this iteration, check point against these.
            nextpoint = false #TODO is there a way to break out of outer for?
            if !isempty(iterminima)
                for z in iterminima
                    if checkrule(p, z, min_distance); nextpoint = true; end
                    nextpoint && continue # skip other minima checks
                end
                nextpoint && continue # skip local search for this point
            end

            # local minimization
            results = fminbox(df, p.x, l, u;xtol=local_xtol, ftol=local_ftol,
                                            grtol=local_grtol, method=method)
            x = results.minimum
            val = results.f_minimum
            f_calls += results.f_calls
            g_calls += results.g_calls
            searches += 1
            converged = results.f_converged || results.gr_converged || 
                        results.x_converged

            if converged
                converges += 1
            
                # Update typical search distance (by rolling average)
                typical_distance = (typical_distance*(searches - 1) + 
                    norm(p.x - x,2)) / searches
                
                # Check if minima already found, if not, add to minimalists
                minfound = false
                for m in minima
                    if norm(x - m.x,2) < distmin
                        minfound = true
                        continue
                    end
                end
                if !minfound #new minima found
            
                    # Update stoplevel
                    stoplevel = exhaustive * doublebox(N) 
            
                    # Update typical minima distance
                    if isempty(minima); min_distance = norm(x - p.x,2); end
                    for m in minima
                       min_distance = min(min_distance, norm(x - m.x,2))
                    end

                    # Gradient not given as output fminbox, needs extra function
                    # evaluation.
                    df.g!(x, g) # no function value required for checkrule
                    g_calls += 1            
                    push!(iterminima, SearchPoint(x, copy(g), val))
                    #Add also to global minima, to check next minima in iteration
                    push!(minima, SearchPoint(x, copy(g), val))
            
                end #if minima found
            end #if converged
        end #for points

        if show_trace 
            @printf "%4d   %3d   %8d   %14d   %6d\n" algo_steps N searches f_calls length(minima)
        end
    end #while

    # Polish off minima
    if polish
        for m in minima
            # run final optization from each found minima
            results = fminbox(df, m.x, l, u;xtol=polish_xtol, ftol=polish_ftol,
                                            grtol=polish_grtol, method=method)
            x = results.minimum
            val = results.f_minimum
            f_calls += results.f_calls
            g_calls += results.g_calls

            # Check if not converges to another final optimization minima
            minfound = false
            for h in polishminina
                if norm(x - h.x,2) < distpolish
                    minfound = true
                    continue
                end
            end
            if !minfound
                df.g!(x, g)
                g_calls += 1
                push!(polishminina, SearchPoint(x, copy(g), val))
            end
        end
        
        if show_trace 
            @printf "Final polish retained %d minima out of %d \n" length(polishminina) length(minima)
        end

        return polishminina, f_calls, g_calls, searches, algo_steps
    else
        return minima, f_calls, g_calls, searches, algo_steps
    end #if polish

end #function

minfinder{T,S}(df::DifferentiableFunction, l::Array{T,1}, u::Array{S,1};kwargs...) = 
    minfinder(df, [convert(Float64, i) for i in l], 
                    [convert(Float64, i) for i in u];kwargs...)
