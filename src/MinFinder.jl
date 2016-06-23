module MinFinder

    using Optim

    #import Optim: optimize

    export optimize, Fminfinder

    "types for the starting points and the minima"
    type SearchPoint{T}
       x::Vector{T} # point in parameter space
       g::Vector{T} # gradient at point (can be nothing)
       val::T       # function value at point
    end
    SearchPoint{T}(x::Vector{T}, g::Vector{T}) = SearchPoint(x, g, nan(T))
    SearchPoint{T}(x::Vector{T}) = SearchPoint(x, Array(T, 0))
    function validpoint(a::SearchPoint, b::SearchPoint, dist)
        #StatsBase.L2dist(a.x, b.x) > dist && dot(a.x - b.x, a.g - b.g) < 0
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
        return sqrt(s) > dist && t < 0
    end
    function validpoint(p::SearchPoint, v::Array{SearchPoint}, dist)
        allvalid = true
        for q in v
            if !validpoint(p, q, dist)
                allvalid = false
                break
            end
        end
        allvalid
    end

    immutable Fminfinder <: Optim.Optimizer; end

    type FminfinderOptimizationResults{T} <: Optim.OptimizationResults
        initial_lower::Array{T}
        initial_upper::Array{T}
        minimum::Array{T}
        f_minimum::Array{T}
        f_calls::Int
        g_calls::Int
        searches::Int
        algo_steps::Int
        N_last::Int
        N_equals_Nmax::Bool
        minima_unpolished::Array{SearchPoint{T}}
        typical_distance::T
        min_distance::T
        converges::Int
        polished::Bool
    end
    iterations(r::FminfinderOptimizationResults) = r.N
    iteration_limit_reached(r::FminfinderOptimizationResults) = r.N_equals_Nmax
    converged(r::FminfinderOptimizationResults) = r.converges > 0
    lower_bound(r::FminfinderOptimizationResults) = r.initial_lower
    upper_bound(r::FminfinderOptimizationResults) = r.initial_upper
    method(r::FminfinderOptimizationResults) = "Fminfinder"


    include("minfinder_main.jl")
    include(joinpath("problems", "multiple_minima.jl"))

end # module
