# Minfinder

[![Build Status](https://travis-ci.org/Ken-B/MinFinder.jl.svg?branch=master)](https://travis-ci.org/Ken-B/MinFinder.jl)

## About

This package originated from a [pull request](https://github.com/JuliaOpt/Optim.jl/pull/73) for the `Optim.jl` package. The idea is to develop this minfinder functionality separate for now until the `Optim.jl` API for bound-constrained optimization is finished. Only julia v0.3 is supported.

I have some ideas for seome extra features, but do let me know in the issues if you have more! For example:
* use low-discrepancy samples for starting point sampling, like from the [`Sobol.jl`](https://github.com/stevengj/Sobol.jl) package
* implement 2 more stopping rules from the MinFinder 2.0 paper, as well as the extra sample checking rule without derivatitves.



## Usage

The [MinFinder algorithm](www.cs.uoi.gr/~lagaris/papers/MINF.pdf) solves those problems where you need to find all the minima for a differentiable function inside a bounded domain. It launches many local optimizations with `fminbox` (in the `Optim.jl` package) from a set of random starting points in the search domain, after some preselection of the points and until a stopping criteria is hit.

Have a good look at [the fminbox section of Optim.jl](https://github.com/JuliaOpt/Optim.jl#box-minimization), because you need to pass your function in a Optim.DifferentiableFunction type.

As an example, consider the Six Hump Camel Back function with 6 minima inside [-5, 5]²:

	camel_f(x) = 4x[1]^2 - 2.1x[1]^4 + 1/3*x[1]^6 + x[1]*x[2] - 4x[2]^2 + 4x[2]^4

	function camel_g!(x,g)
	    g[1] = 8x[1] - 8.4x[1]^3 + 2x[1]^5 + x[2]
	    g[2] = x[1] - 8x[2] + 16x[2]^3
	    return nothing
	end

	function camel_fg!(x, g)
	    g[1] = 8x[1] - 8.4x[1]^3 + 2x[1]^5 + x[2]
	    g[2] = x[1] - 8x[2] + 16x[2]^3
	    return 4x[1]^2 - 2.1x[1]^4 + 1/3*x[1]^6 + x[1]*x[2] - 4x[2]^2 + 4x[2]^4
	end	

	df = Optim.DifferentiableFunction(camel_f,camel_g!,camel_fg!)

    minima, f_calls, g_calls, searches, steps = minfinder(camel, [-5, -5], [5, 5]; show_trace=true)

The output `minima` is a vector with elements of type `SearchPoint`, each with fields `x` for location, `f` for function value and `g` for gradient.

Additional options are:
* `NINIT`: initial number of sample points in the search space per iteration (default = 20).
* `NMAX`: maximum number of sample points in the search space per iteration (default = 250, the 100 of the paper is too small).
* `ENRICH`: multiplication of sample points N when more than half of sample points failed preselection criteria (default = 1.1).
* `EXHAUSTIVE`: in (0,1). For small values of p (p→0) the algorithm searches the area exhaustively, while for p→1, the algorithm terminates earlier, but perhaps prematurely (default = 0.5).
* `max_iter`: maximum number of iterations, each with N ponts samples and local searches (default = 10_000).
* `show_trace`: print iteration results when `show_iter > 0` (default = 0)
* `distmin`: the results of a local search will be added to the `minima` list if its location differs less than this threshold from previously found minima. Increase when lots of returned minima correspond to the same physical point (default = `sqrt(local_tol)`).
* `distpolish`: same as distmin for final polish phase (default = `sqrt(polish_tol)` ).
* `polish`: boolean flag to indicate whether to perform an extra search at the very end for each minima found, to polish off the found minima with extra precision (default = true).
* `local_tol`: tolerance level passed to fminbox for the local searches (default = `eps(Type)^(2/3)` or `sqrt(eps(T)^(2/3))` when `polish=true`).
* `polish_tol`: tolerance level of final polish searches (default = `eps(Type)^(2/3)`).
