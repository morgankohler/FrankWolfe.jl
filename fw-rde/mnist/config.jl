indices = 0:49
rates = [784]
d = 25
max_iter = 1000

fw_arguments = (
    line_search=FrankWolfe.MonotonousNonConvexStepSize(),
    max_iteration=max_iter,
    print_iter=max_iter / 10,
    verbose=true,
    #lazy=true,
)
