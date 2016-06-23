module MultipleMinimaProblems

# Test problems with multiple minima from MinFinder paper and from:
# http://www2.compute.dtu.dk/~kajm/Test_ex_forms/test_ex.html
using Optim #for DifferentiableFunction

immutable OptimizationProblem
    name::ASCIIString
    f::Optim.DifferentiableFunction
    l::Vector
    u::Vector
    minima # Vector with all local minima
    # glob_x::Vector{Float64} # set of global minimum points
    glob_f::AbstractFloat # global minimum function value
end

examples = Dict{ASCIIString, OptimizationProblem}()

## Rosenbrock ##
examples["Rosenbrock"] = OptimizationProblem(
    "Rosenbrock",
    Optim.DifferentiableFunction(
        x -> (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2,
        (x, g) -> begin
            g[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
            g[2] = 200.0 * (x[2] - x[1]^2)
        end,
        (x, g) -> begin
            d1 = (1.0 - x[1])
            d2 = (x[2] - x[1]^2)
            g[1] = -2.0 * d1 - 400.0 * d2 * x[1]
            g[2] = 200.0 * d2
            return d1^2 + 100.0 * d2^2
        end),
    [-2., -2],
    [2.,2],
    Vector[[1.,1.]],
    0.)

## Camel ##
examples["Camel"] = OptimizationProblem(
    "Camel",
    Optim.DifferentiableFunction(
        x -> 4x[1]^2 - 2.1x[1]^4 + 1/3*x[1]^6 + x[1]*x[2] - 4x[2]^2 + 4x[2]^4,
        function camel_g!(x,g)
            g[1] = 8x[1] - 8.4x[1]^3 + 2x[1]^5 + x[2]
            g[2] = x[1] - 8x[2] + 16x[2]^3
            return nothing
        end,
        function camel_fg!(x, g)
            g[1] = 8x[1] - 8.4x[1]^3 + 2x[1]^5 + x[2]
            g[2] = x[1] - 8x[2] + 16x[2]^3
            return 4x[1]^2 - 2.1x[1]^4 + 1/3*x[1]^6 + x[1]*x[2] - 4x[2]^2 + 4x[2]^4
        end),
    [-5., -5],
    [5., 5],
    Vector[[-0.0898416,0.712656],[0.089839,-0.712656], [-1.6071,-0.568651],
       [1.70361,-0.796084], [1.6071,0.568652], [-1.70361,0.796084]],
    -1.03163)

## Rastrigin ##
examples["Rastrigin"] = OptimizationProblem(
    "Rastrigin",
    Optim.DifferentiableFunction(
        x -> x[1]^2 + x[2]^2 - cos(18x[1]) - cos(18x[2]),
        (x,g) -> begin
            g[1] = 2x[1] +18sin(18x[1])
            g[2] = 2x[2] +18sin(18x[2])
            return nothing
        end,
        (x,g) -> begin
            g[1] = 2x[1] +18sin(18x[1])
            g[2] = 2x[2] +18sin(18x[2])
            return x[1]^2 + x[2]^2 - cos(18x[1]) - cos(18x[2])
        end),
    [-1., -1],
    [1., 1],
    Vector[[0.0,0.0],[0.34692,0.0],[-0.34693,0.0],[0.69384,0.0],[-0.69385,0.0],
     [0.99999,0.0],[-1.0,0.0],[0.0,0.34692],[0.34692,0.34692],
     [-0.34693,0.34692],[0.69384,0.34692],[-0.69385,0.34692],[0.99999,0.34692],
     [-1.0,0.34692],[0.0,-0.34693],[0.34692,-0.34693],[-0.34693,-0.34693],
     [0.69384,-0.34693],[-0.69385,-0.34693],[0.99999,-0.34693],[-1.0,-0.34693],
     [0.0,0.69384],[0.34692,0.69384],[-0.34693,0.69384],[0.69384,0.69384],
     [-0.69385,0.69384],[0.99999,0.69384],[-1.0,0.69384],[0.0,-0.69385],
     [0.34692,-0.69385],[-0.34693,-0.69385],[0.69384,-0.69385],
     [-0.69385,-0.69385],[0.99999,-0.69385],[-1.0,-0.69385],[0.0,0.99999],
     [0.34692,0.99999],[-0.34693,0.99999],[0.69384,0.99999],[-0.69385,0.99999],
     [0.99999,0.99999],[-1.0,0.99999],[0.0,-1.0],[0.34692,-1.0],[-0.34693,-1.0],
     [0.69384,-1.0],[-0.69385,-1.0],[0.99999,-1.0],[-1.0,-1.0]],
    -2.)

## Shekel ##

# can take parameter m = 5, 7 or 10
const shekel_A =[4 4 4 4;
                 1 1 1 1;
                 8 8 8 8;
                 6 6 6 6;
                 3 7 3 7;
                 2 9 2 9;
                 5 5 3 3;
                 8 1 8 1;
                 6 2 6 2;
                 7 3.6 7 3.6]
const shekel_c = [.1 .2 .2 .4 .4 .6 .3 .7 .5 .5]

shekel(x;m=5)=-sum([1/(norm(x - vec(shekel_A[i,:]), 2)^2 + shekel_c[i]) for i=1:m])
function shekel_g!(x::Vector, g; m=5)
    #length(x) == 4 || error("input needs to be 4-dimensional")
    for j = 1:4
        g[j] = sum([2*(x[j] - shekel_A[i,j]) /
            (norm(x - vec(shekel_A[i,:]),2)^2 + shekel_c[i])^2 for i=1:m])
    end
    return nothing
end
function shekel_fg!(x::Vector, g; m=5)
    #length(x) == 4 || error("input needs to be 4-dimensional")
    for j = 1:4
        g[j] = sum([2*(x[j] - shekel_A[i,j]) /
            (norm(x - vec(shekel_A[i,:]),2)^2 + shekel_c[i])^2 for i=1:m])
    end
    return -sum([1/(norm(x - vec(shekel_A[i,:]),2)^2 + shekel_c[i]) for i=1:m])
end

examples["Shekel5"] = OptimizationProblem(
    "Shekel5",
    Optim.DifferentiableFunction(shekel,shekel_g!,shekel_fg!),
    zeros(4),
    10*ones(4),
    Vector[[4.00003,4.00013,4.00003,4.00013],[1.00013,1.00015,1.00013,1.00015],
     [5.99874,6.00028,5.99874,6.00028],[3.00179,6.99833,3.00179,6.99833],
     [7.99958,7.99964,7.99958,7.99964]],
    -10.1532)

examples["Shekel7"] = OptimizationProblem(
    "Shekel7",
    Optim.DifferentiableFunction(x->shekel(x;m=7),(x,g)->shekel_g!(x,g;m=7),
        (x,g)->shekel_fg!(x,g;m=7)),
    zeros(4),
    10*ones(4),
    Vector[[4.00057,4.00068,3.99948,3.9996],[1.00023,1.00027,1.00018,1.00022],
     [3.0009,7.00064,3.00036,7.0001],[5.9981,6.00008,5.99732,5.9993],
     [4.99422,4.99499,3.00606,3.00682],[2.0048,8.99168,2.00462,8.99149],
     [7.99951,7.99962,7.99949,7.9996]],
    -10.4029)

examples["Shekel10"] = OptimizationProblem(
    "Shekel10",
    Optim.DifferentiableFunction(x->shekel(x;m=10),(x,g)->shekel_g!(x,g;m=10),
        (x,g)->shekel_fg!(x,g;m=10)),
    zeros(4),
    10*ones(4),
    Vector[[1.00036,1.0003,1.00031,1.00025],[3.00127,7.00022,3.00073,6.99968],
     [6.00557,2.01001,6.00437,2.0088],[5.99901,5.99728,5.99823,5.9965],
     [4.00074,4.00059,3.99966,3.9995],[7.99947,7.99945,7.99946,7.99943],
     [4.99487,4.99398,3.00755,3.00666],[6.99163,3.59557,6.99065,3.5946],
     [2.0051,8.99129,2.00491,8.9911],[7.98677,1.01223,7.98644,1.0119]],
     -10.5364)

end # module
