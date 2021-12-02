# using Pkg
# Pkg.activate(@__DIR__)
using Pkg
Pkg.activate("/home/Morgan/FrankWolfe.jl")

# parse command line arguments if given
if length(ARGS) > 0
    subdir = ARGS[1]
# otherwise prompt user to specify
else
    print("Please enter sub directory to run RDE in: ")
    subdir = readline()
end
# input validation
while !isdir(joinpath(@__DIR__, subdir))
    print("Invalid directory $subdir. Please enter sub directory to run RDE in: ")
    global subdir = readline()
end

using PyCall
pushfirst!(PyVector(pyimport("sys")["path"]), joinpath(@__DIR__, subdir))

import FrankWolfe
include("custom_oralces.jl")
include(joinpath(@__DIR__, subdir, "config.jl"))  # load indices, rates, max_iter
cd(subdir)

# Get the Python side of RDE
rde = pyimport("rde")

for idx in indices

    # Load data sample and distortion functional
    x, fname = rde.get_data_sample(idx)
    rde.store_test(x, "untargeted_ktest")

    f, df, node, pred = rde.get_distortion(x)

    # Wrap objective and gradiet functions
    function func(s)
        if !(s isa Vector{eltype(x)})
            s = convert(Vector{eltype(x)}, s)
        end
        return f(s)
    end

    function grad!(storage, s)
        if !(s isa Vector{eltype(x)})
            s = convert(Vector{eltype(x)}, s)
        end
        g = df(s)
        return @. storage = g
    end

    all_s = zeros(eltype(x), (length(rates), length(x)))

    # rates = [50, 100, 150, 200, 250, 300, 350, 400, 1000, 2000]

    for rate in rates
        print("\nrate: ")
        print(rate)
        print("\n")
        # Run FrankWolfe
        println("Running sample $idx with rate $rate")
        s0 = similar(x[:])
        s0 .= 0.0
        lmo = NonNegKSparseLMO(rate, 1.0)

        @time s, v, primal, dual_gap = FrankWolfe.frank_wolfe(
        #@time s, v, primal, dual_gap = FrankWolfe.away_frank_wolfe(
        #@time s, v, primal, dual_gap = FrankWolfe.blended_conditional_gradient(
        #@time s, v, primal, dual_gap = FrankWolfe.lazified_conditional_gradient(
            s -> func(s),
            (storage, s) -> grad!(storage, s),
            lmo,
            s0,
            ;fw_arguments...
        )
        # reset adaptive step size if necessary
        if fw_arguments.line_search isa FrankWolfe.MonotonousNonConvexStepSize
            fw_arguments.line_search.factor = 0
        end

        # Store single rate result
        all_s[indexin(rate, rates)[1], :] = s
        # rde.store_single_result(s, idx, fname, rate)
        rde.store_single_result(s, idx, "untargeted_ktest", rate)

        # rde.store_test(s, "untargeted_test")
        
    end

    break

    # Store multiple rate results
    rde.store_collected_results(all_s, idx, node, pred, fname, rates)

end
