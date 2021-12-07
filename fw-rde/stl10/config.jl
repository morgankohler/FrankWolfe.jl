using Random
using Ranges

indices = randperm(MersenneTwister(1234), 8000)[1:20]
# rates = [2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000, 34000, 38000, 42000, 46000, 50000]
rates = [8000]
d_range = [0.1, 0.5, 1, 2, 4, 8, 12]
d_range = convert(Array{Float64,1}, d_range)
max_iter = 10
mode = "untargeted"  # "untargeted" or "targeted"
optim = "joint"  # "univariate" or "joint"
save_imp = false
save_normacc_graph = true
test_name = "joint_norm_acc"  # univariate_norm_acc

fw_arguments = (
    line_search=FrankWolfe.MonotonousNonConvexStepSize(),
    max_iteration=max_iter,
    print_iter=max_iter / 10,
    verbose=true,
    #lazy=true,
)
