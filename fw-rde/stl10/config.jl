indices = 0:50  # 3786 #  0:49 #0:7999
# rates = [2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000, 34000, 38000, 42000, 46000, 50000]
rates = [8000]
d = 12.75
max_iter = 40
mode = "targeted"
optim = "joint" #"univariate"
save_imp = false
test_name = "univariate_untargeted_test"

fw_arguments = (
    line_search=FrankWolfe.MonotonousNonConvexStepSize(),
    max_iteration=max_iter,
    print_iter=max_iter / 10,
    verbose=true,
    #lazy=true,
)
