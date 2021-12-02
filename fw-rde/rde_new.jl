using Pkg
Pkg.activate("/home/Morgan/FrankWolfe.jl")
# Pkg.activate("/home/Morgan/FrankWolfe.jl/fw-rde")



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
    print("Invalid directory $subdir. Please enter sub directory to run RDE in:")
    global subdir = readline()
end

using PyCall
pushfirst!(PyVector(pyimport("sys")["path"]), joinpath(@__DIR__, subdir))

import FrankWolfe
import FrankWolfe: LpNormLMO
include("custom_oralces.jl")
include(joinpath(@__DIR__, subdir, "config.jl"))  # load indices, rates, max_iter
cd(subdir)

# Get the Python side of RDE
rde = pyimport("rde_new")

for idx in indices

    # Load data sample and distortion functional
    x, fname = rde.get_data_sample(idx)
#     rde.store_test(x, "untargeted_ktest")
    rde.store_single_result(x, "xb4", "untargeted_2var_testd25", 0)

#     print(x)
#     throw(ErrorException)

    f, df_s, df_p, node, pred = rde.get_distortion(x)

    # Wrap objective and gradiet functions
    function func(s, p)
        if !(s isa Vector{eltype(x)})
            s = convert(Vector{eltype(x)}, s)
        end
        if !(p isa Vector{eltype(x)})
            p = convert(Vector{eltype(x)}, p)
        end
        return f(s, p)
    end

    function grad_s!(storage, s, p)
        if !(s isa Vector{eltype(x)})
            s = convert(Vector{eltype(x)}, s)
        end
        if !(p isa Vector{eltype(x)})
            p = convert(Vector{eltype(x)}, p)
        end
        g = df_s(s, p)
        return @. storage = g
    end

    function grad_p!(storage, s, p)
        if !(s isa Vector{eltype(x)})
            s = convert(Vector{eltype(x)}, s)
        end
        if !(p isa Vector{eltype(x)})
            p = convert(Vector{eltype(x)}, p)
        end
        g = df_p(s, p)
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
        lmo_s = NonNegKSparseLMO(rate, 1.0)

        p0 = similar(x[:])
        p0 .= 0.0
        lmo_p = LpNormLMO{Float64,Inf}(d)

        @time s, v_s, p, v_p, primal, dual_gap_s, dual_gap_p  = FrankWolfe.frank_wolfe_2var(
        #@time s, v, primal, dual_gap = FrankWolfe.frank_wolfe(
        #@time s, v, primal, dual_gap = FrankWolfe.away_frank_wolfe(
        #@time s, v, primal, dual_gap = FrankWolfe.blended_conditional_gradient(
        #@time s, v, primal, dual_gap = FrankWolfe.lazified_conditional_gradient(
            (s, p) -> func(s, p),
            (storage, s, p) -> grad_s!(storage, s, p),
            (storage, s, p) -> grad_p!(storage, s, p),
            lmo_s,
            lmo_p,
            s0,
            p0,
            ;fw_arguments...
        )
        # reset adaptive step size if necessary
        if fw_arguments.line_search isa FrankWolfe.MonotonousNonConvexStepSize
            fw_arguments.line_search.factor = 0
        end

        rde.print_model_prediction(x, s, p)

        # Store single rate result
        all_s[indexin(rate, rates)[1], :] = s
        # rde.store_single_result(s, idx, fname, rate)
        rde.store_single_result(s, "s", "untargeted_2var_testd25", rate)
        rde.store_single_result(p, "p", "untargeted_2var_testd25", rate)
        rde.store_single_result(x, "x", "untargeted_2var_testd25", rate)
        rde.store_pert_img(x, s, p, "pertimg", "untargeted_2var_testd25", rate)

        # rde.store_test(s, "untargeted_test")
        
    end

    break

    # Store multiple rate results
    rde.store_collected_results(all_s, idx, node, pred, fname, rates)

end
