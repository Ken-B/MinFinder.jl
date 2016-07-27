# Minfinder

[![Build Status](https://travis-ci.org/Ken-B/MinFinder.jl.svg?branch=master)](https://travis-ci.org/Ken-B/MinFinder.jl)

The [MinFinder algorithm](www.cs.uoi.gr/~lagaris/papers/MINF.pdf) solves those problems where you need to find all the minima for a differentiable function inside a bounded domain. It launches many local optimizations from a set of random starting points in the search domain, after some preselection of the points and until a stopping criteria is hit. The local searches use the `Fminbox` method from the `Optim.jl` package.

## About

This package originated from a [pull request](https://github.com/JuliaOpt/Optim.jl/pull/73) for the `Optim.jl` package but now simply extends that package.

I have some ideas for some extra features, but do let me know in the issues if you have more. For example:

* use low-discrepancy samples for the starting points from the [`Sobol.jl`](https://github.com/stevengj/Sobol.jl) package
* implement 2 more stopping rules from the MinFinder 2.0 paper, as well as the extra sample checking rule without derivatives.

## Usage

Have a good look at [the `Fminbox` section of Optim.jl](https://github.com/JuliaOpt/Optim.jl#box-minimization), because you need to pass your function in a Optim.DifferentiableFunction type.

As an example, consider the Six Hump Camel Back function with 6 minima inside [-5, 5]²:

	camel_f(x) = 4x[1]^2 - 2.1x[1]^4 + 1/3*x[1]^6 + x[1]*x[2] - 4x[2]^2 + 4x[2]^4

	function camel_g!(x, g) #gradient evaluation to pre-allocated array
	    g[1] = 8x[1] - 8.4x[1]^3 + 2x[1]^5 + x[2]
	    g[2] = x[1] - 8x[2] + 16x[2]^3
	    return nothing
	end

	function camel_fg!(x, g) #function call and gradient combined
	    g[1] = 8x[1] - 8.4x[1]^3 + 2x[1]^5 + x[2]
	    g[2] = x[1] - 8x[2] + 16x[2]^3
	    return 4x[1]^2 - 2.1x[1]^4 + 1/3*x[1]^6 + x[1]*x[2] - 4x[2]^2 + 4x[2]^4
	end

	df = Optim.DifferentiableFunction(camel_f, camel_g!, camel_fg!)

    result = optimize(df, [-5, -5], [5, 5]; show_trace=true)

The output is of type `FminfinderOptimizationResults` with following methods defined:

* `Optim.minimizer`: vector of points (each again a vector) where function reaches a minimum
* `Optim.minimum`: vector of function values at those points
* `Optim.converged`: whether a local search has converged at least once
* `Optim.lower_bound`, `Optim.upper_bound`, `Optim.method`, `Optim.f_calls`, `Optim.g_calls` : *self explanatory*



Additional options are:

* `Ninit`: initial number of sample points in the search space per iteration (default 20).
* `Nmax`: maximum number of sample points in the search space per iteration (default 100 as in the paper, but I usually find 250 more appropriate).
* `enrich`: multiplication of sample points N when more than half of sample points failed preselection criteria (default= 1.1).
* `exhaustive`: in (0,1). For small values p→0 the algorithm searches the area exhaustively, while for p→1 the algorithm terminates earlier, but perhaps prematurely (default value 0.5).
* `max_algo_steps`: maximum number of iterations, each with N points samples and local searches (default 10_000).
* `show_trace`: print iteration results (default = true)
* `dist_unique`: the results of a local search will be added to the `minima` list if its location differs less than this threshold from previously found minima. Increase when lots of returned minima correspond to the same physical point (default = `sqrt(local_tol)`).
* `polish`: boolean flag to indicate whether to perform an extra search at the very end for each minima found, to polish off the found minima with extra precision (default = true).
* `distpolish`: same as `distmin` for final polish phase (default `sqrt(polish_tol)` ).
* `local_xtol`: tolerance level used for the local searches, default eps(Type) (or `sqrt(eps(T)^(2/3))` when `polish=true`). Similar poslish_ftol and polish_gtol (default `eps(Type)^(2/3)`, or `sqrt(eps(T)^(2/3))` when `polish=true`)
* `polish_xtol`: tolerance level of final polish searches (default `eps(Type)^(2/3)`).
