
"""
    frank_wolfe(f, grad!, lmo, x0; ...)

Simplest form of the Frank-Wolfe algorithm.
Returns a tuple `(x, v, primal, dual_gap, traj_data)` with:
- `x` final iterate
- `v` last vertex from the LMO
- `primal` primal value `f(x)`
- `dual_gap` final Frank-Wolfe gap
- `traj_data` vector of trajectory information.
"""
function frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    line_search::LineSearchMethod=Adaptive(),
    L=Inf,
    gamma0=0,
    step_lim=20,
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis::Emphasis=memory,
    gradient=nothing,
    callback=nothing,
    timeout=Inf,
    print_callback=print_callback,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    tt = regular
    traj_data = []
    if trajectory && callback === nothing
        callback = trajectory_callback(traj_data)
    end
    time_start = time_ns()

    if line_search isa Shortstep && !isfinite(L)
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search isa FixedStep && gamma0 == 0
        println("FATAL: gamma0 not set. We are not going to move a single bit.")
    end

    if (!isnothing(momentum) && line_search isa Union{Shortstep,Adaptive,RationalShortstep})
        println(
            "WARNING: Momentum-averaged gradients should usually be used with agnostic stepsize rules.",
        )
    end

    if verbose
        println("\nVanilla Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $numType",
        )
        grad_type = typeof(gradient)
        println("MOMENTUM: $momentum GRADIENTTYPE: $grad_type")
        if emphasis === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec"]
        print_callback(headers, format_string, print_header=true)
    end
    if emphasis == memory && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end
    first_iter = true
    # instanciating container for gradient
    if gradient === nothing
        gradient = similar(x)
    end

    # container for direction
    d = similar(x)
    gtemp = if momentum === nothing
        nothing
    else
        similar(x)
    end
    while t <= max_iteration && dual_gap >= max(epsilon, eps())

        #####################
        # managing time and Ctrl-C
        #####################
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

        #####################


        if momentum === nothing || first_iter
            grad!(gradient, x)
            if momentum !== nothing
                gtemp .= gradient
            end
        else
            grad!(gtemp, x)
            @emphasis(emphasis, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
        end
        first_iter = false

        v = compute_extreme_point(lmo, gradient)
        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            line_search isa Shortstep
        )
            primal = f(x)
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        end

        @emphasis(emphasis, d = x - v)

        gamma, L = line_search_wrapper(
            line_search,
            t,
            f,
            grad!,
            x,
            d,
            momentum === nothing ? gradient : gtemp, # use appropriate storage
            dual_gap,
            L,
            gamma0,
            linesearch_tol,
            step_lim,
            one(eltype(x)),
        )
        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap,
                dual_gap=dual_gap,
                time=tot_time,
                x=x,
                v=v,
                gamma=gamma,
            )
            callback(state)
        end

        @emphasis(emphasis, x = x - gamma * d)

        if (mod(t, print_iter) == 0 && verbose)
            tt = regular
            if t == 0
                tt = initial
            end

            rep = (
                st[Symbol(tt)],
                string(t),
                Float64(primal),
                Float64(primal - dual_gap),
                Float64(dual_gap),
                tot_time,
                t / tot_time,
            )
            print_callback(rep, format_string)

            flush(stdout)
        end
        t = t + 1
    end
    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.

    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = last
        tot_time = (time_ns() - time_start) / 1.0e9
        rep = (
            st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            tot_time,
            t / tot_time,
        )
        print_callback(rep, format_string)
        print_callback(nothing, format_string, print_footer=true)
        flush(stdout)
    end
    return x, v, primal, dual_gap, traj_data
end


function frank_wolfe_2var(
    f,
    grad!,
    lmo_x,
    lmo_p,
    x0,  # this is s0
    p0;
    line_search::LineSearchMethod=Adaptive(),
    L=Inf,
    gamma0=0,
    step_lim=20,
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis::Emphasis=memory,
    gradient=nothing,
    callback=nothing,
    timeout=Inf,
    print_callback=print_callback,
    )

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
    gamma0_x = 0
    gamma0_p = 0
    L_x = Inf
    L_p = Inf
    t = 0
    dual_gap_x = Inf
    dual_gap_p = Inf
    primal = Inf
    v_x = []
    v_p = []
    x = x0
    p = p0
    tt = regular
    traj_data = []
    if trajectory && callback === nothing
        callback = trajectory_callback(traj_data)
    end
    time_start = time_ns()

    if line_search isa Shortstep && !isfinite(L)
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search isa FixedStep && gamma0 == 0
        println("FATAL: gamma0 not set. We are not going to move a single bit.")
    end

    if (!isnothing(momentum) && line_search isa Union{Shortstep,Adaptive,RationalShortstep})
        println(
            "WARNING: Momentum-averaged gradients should usually be used with agnostic stepsize rules.",
        )
    end

    if verbose
        println("\nVanilla Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $numType",
        )
        grad_type = typeof(gradient)
        println("MOMENTUM: $momentum GRADIENTTYPE: $grad_type")
        if emphasis === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec"]
        print_callback(headers, format_string, print_header=true)
    end
    if emphasis == memory && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end
    if emphasis == memory && !isa(p, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(p) <: Integer
            p = copyto!(similar(p, float(eltype(p))), p)
        else
            p = copyto!(similar(p), p)
        end
    end
    first_iter = true
    # instanciating container for gradient

    gradient_x = similar(x)
    gradient_p = similar(x)

    # container for direction
    d_x = similar(x)
    d_p = similar(x)
    gtemp_x = similar(x)
    gtemp_p = similar(x)

    gtemp = Nothing
    momentum_x = Nothing
    momentum_p = Nothing

    while t <= max_iteration # && dual_gap >= max(epsilon, eps())

        #####################
        # managing time and Ctrl-C
        #####################
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

        #####################


        if momentum_x === nothing || first_iter
            grad!(gradient, x, p)
            gradient_x = gradient[1]
            gradient_p = gradient[2]
            if momentum_x !== nothing
                gtemp_x .= gradient[1]
                gtemp_p .= gradient[2]
            end
        else
            grad!(gtemp, x, p)
            gtemp_x = gtemp[1]
            gtemp_p = gtemp[2]
            @emphasis(emphasis, gradient_x = (momentum_x * gradient_x) + (1 - momentum_x) * gtemp_x)
            @emphasis(emphasis, gradient_p = (momentum_p * gradient_p) + (1 - momentum_p) * gtemp_p)
        end
        first_iter = false

        v_x = compute_extreme_point(lmo_x, gradient_x)
        v_p = compute_extreme_point(lmo_p, gradient_p)
        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            line_search isa Shortstep
        )
            primal = f(x, p)
            dual_gap_x = fast_dot(x, gradient_x) - fast_dot(v_x, gradient_x)
            dual_gap_p = fast_dot(p, gradient_p) - fast_dot(v_p, gradient_p)
        end

        @emphasis(emphasis, d_x = x - v_x)
        @emphasis(emphasis, d_p = p - v_p)

        function g_x(grad, x_local)
            grad_local = Nothing
            grad!(grad_local, x_local, p)
            return @. grad = grad_local[1]
        end

        function g_p(grad, p_local)
            grad_local = Nothing
            grad!(grad_local, x, p_local)
            return @. grad = grad_local[2]
        end

        function f_x(x_local)
            return f(x_local, p)
        end

        function f_x(p_local)
            return f(x, p_local)
        end

#         f_x = (x_local) -> f(x_local, p)
#         f_p = (p_local) -> f(x, p_local)
#         g_x = (grad_local, x_local) -> grad!(grad_local, x_local, p)
#         g_p = (grad_local, p_local) -> grad!(grad_local, x, p_local)

        gamma_x, L_x = line_search_wrapper(
            line_search,
            t,
            f_x,
            g_x,
            x,
            d_x,
            momentum_x === nothing ? gradient_x : gtemp_x, # use appropriate storage
            dual_gap_x,
            L_x,
            gamma0_x,
            linesearch_tol,
            step_lim,
            one(eltype(x)),
        )
        gamma_p, L_p = line_search_wrapper(
            line_search,
            t,
            f_p,
            g_p,
            p,
            d_p,
            momentum_p === nothing ? gradient_p : gtemp_p, # use appropriate storage
            dual_gap_p,
            L_p,
            gamma0_p,
            linesearch_tol,
            step_lim,
            one(eltype(p)),
        )

        if callback !== nothing
            state_x = (
                t=t,
                primal=primal,
                dual=primal - dual_gap_x,
                dual_gap=dual_gap_p,
                time=tot_time,
                x=x,
                v=v_x,
                gamma=gamma_x,
            )
            callback(state)

            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap_p,
                dual_gap=dual_gap_p,
                time=tot_time,
                x=p,
                v=v_p,
                gamma=gamma_p,
            )
            callback(state)

        end

        @emphasis(emphasis, x = x - gamma_x * d_x)
        @emphasis(emphasis, p = p - gamma_p * d_p)

        if (mod(t, print_iter) == 0 && verbose)
            tt = regular
            if t == 0
                tt = initial
            end

            rep = (
                st[Symbol(tt)],
                string(t),
                Float64(primal),
                Float64(primal - dual_gap_p),
                Float64(dual_gap_p),
                tot_time,
                t / tot_time,
            )
            print_callback(rep, format_string)

            flush(stdout)
        end
        t = t + 1
    end
    # recompute everything once for final verification / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.

#     grad!(gradient, x)
#     v = compute_extreme_point(lmo, gradient)
#     primal = f(x)
#     dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
#     if verbose
#         tt = last
#         tot_time = (time_ns() - time_start) / 1.0e9
#         rep = (
#             st[Symbol(tt)],
#             string(t - 1),
#             Float64(primal),
#             Float64(primal - dual_gap),
#             Float64(dual_gap),
#             tot_time,
#             t / tot_time,
#         )
#         print_callback(rep, format_string)
#         print_callback(nothing, format_string, print_footer=true)
#         flush(stdout)
#     end
    return x, v_x, p, v_p, primal, dual_gap_x, dual_gap_p, traj_data
end


"""
    lazified_conditional_gradient

Similar to [`frank_wolfe`](@ref) but lazyfying the LMO:
each call is stored in a cache, which is looked up first for a good-enough direction.
The cache used is a [`FrankWolfe.MultiCacheLMO`](@ref) or a [`FrankWolfe.VectorCacheLMO`](@ref)
depending on whether the provided `cache_size` option is finite.
"""
function lazified_conditional_gradient(
    f,
    grad!,
    lmo_base,
    x0;
    line_search::LineSearchMethod=Adaptive(),
    L=Inf,
    gamma0=0,
    K=2.0,
    cache_size=Inf,
    greedy_lazy=false,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    step_lim=20,
    emphasis::Emphasis=memory,
    gradient=nothing,
    VType=typeof(x0),
    callback=nothing,
    timeout=Inf,
    print_callback=print_callback,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i\n"

    if isfinite(cache_size)
        lmo = MultiCacheLMO{cache_size,typeof(lmo_base),VType}(lmo_base)
    else
        lmo = VectorCacheLMO{typeof(lmo_base),VType}(lmo_base)
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    phi = Inf
    traj_data = []
    if trajectory && callback === nothing
        callback = trajectory_callback(traj_data)
    end
    tt = regular
    time_start = time_ns()

    if line_search isa Shortstep && !isfinite(L)
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search isa Agnostic || line_search isa Nonconvex
        println("FATAL: Lazification is not known to converge with open-loop step size strategies.")
    end

    if verbose
        println("\nLazified Conditional Gradients (Frank-Wolfe + Lazification).")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration K: $K TYPE: $numType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type CACHESIZE $cache_size GREEDYCACHE: $greedy_lazy")
        if emphasis == memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers =
            ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "Cache Size"]
        print_callback(headers, format_string, print_header=true)
    end

    if emphasis == memory && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end

    if gradient === nothing
        gradient = similar(x)
    end

    # container for direction
    d = similar(x)

    while t <= max_iteration && dual_gap >= max(epsilon, eps(float(eltype(x))))

        #####################
        # managing time and Ctrl-C
        #####################
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

        #####################

        grad!(gradient, x)

        threshold = fast_dot(x, gradient) - phi / K

        # go easy on the memory - only compute if really needed
        if ((mod(t, print_iter) == 0 && verbose ) || callback !== nothing)
            primal = f(x)
        end

        v = compute_extreme_point(lmo, gradient, threshold=threshold, greedy=greedy_lazy)
        tt = lazy
        if fast_dot(v, gradient) > threshold
            tt = dualstep
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
            phi = min(dual_gap, phi / 2)
        end

        @emphasis(emphasis, d = x - v)

        gamma, L = line_search_wrapper(
            line_search,
            t,
            f,
            grad!,
            x,
            d,
            gradient,
            dual_gap,
            L,
            gamma0,
            linesearch_tol,
            step_lim,
            1.0,
        )

        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap,
                dual_gap=dual_gap,
                time=tot_time,
                cache_size=length(lmo),
                x=x,
                v=v,
                gamma=gamma
            )
            callback(state)
        end

        @emphasis(emphasis, x = x - gamma * d)

        if verbose && (mod(t, print_iter) == 0 || tt == dualstep)
            if t == 0
                tt = initial
            end
            rep = (
                st[Symbol(tt)],
                string(t),
                Float64(primal),
                Float64(primal - dual_gap),
                Float64(dual_gap),
                tot_time,
                t / tot_time,
                length(lmo),
            )
            print_callback(rep, format_string)
            flush(stdout)
        end
        t += 1
    end

    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)

    if verbose 
        tt = last
        tot_time = (time_ns() - time_start) / 1.0e9
        rep = (
            st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            tot_time,
            t / tot_time,
            length(lmo),
        )
        print_callback(rep, format_string)
        print_callback(nothing, format_string, print_footer=true)
        flush(stdout)
    end
    return x, v, primal, dual_gap, traj_data
end

"""
    stochastic_frank_wolfe(f::StochasticObjective, lmo, x0; ...)

Stochastic version of Frank-Wolfe, evaluates the objective and gradient stochastically,
implemented through the [FrankWolfe.StochasticObjective](@ref) interface.

Keyword arguments include `batch_size` to pass a fixed `batch_size`
or a `batch_iterator` implementing
`batch_size = FrankWolfe.batchsize_iterate(batch_iterator)` for algorithms like
Variance-reduced and projection-free stochastic optimization, E Hazan, H Luo, 2016.

Similarly, a constant `momentum` can be passed or replaced by a `momentum_iterator`
implementing `momentum = FrankWolfe.momentum_iterate(momentum_iterator)`.
"""
function stochastic_frank_wolfe(
    f::StochasticObjective,
    lmo,
    x0;
    line_search::LineSearchMethod=Nonconvex(),
    L=Inf,
    gamma0=0,
    momentum_iterator=nothing,
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis::Emphasis=memory,
    rng=Random.GLOBAL_RNG,
    batch_size=length(f.xs) ÷ 10 + 1,
    batch_iterator=nothing,
    full_evaluation=false,
    callback=nothing,
    timeout=Inf,
    print_callback=print_callback,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %6i\n"

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    tt = regular
    traj_data = []
    if trajectory && callback === nothing
        callback = trajectory_callback(traj_data)
    end
    time_start = time_ns()

    if line_search == Shortstep && L == Inf
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search == FixedStep && gamma0 == 0
        println("FATAL: gamma0 not set. We are not going to move a single bit.")
    end
    if momentum_iterator === nothing && momentum !== nothing
        momentum_iterator = ConstantMomentumIterator(momentum)
    end
    if batch_iterator === nothing
        batch_iterator = ConstantBatchIterator(batch_size)
    end

    if verbose
        println("\nStochastic Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration TYPE: $numType",
        )
        println("GRADIENTTYPE: $(typeof(f.storage)) MOMENTUM: $(momentum_iterator !== nothing) batch policy: $(typeof(batch_iterator)) ")
        if emphasis == memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "batch size")
        print_callback(headers, format_string, print_header=true)
    end

    if emphasis == memory && !isa(x, Union{Array, SparseArrays.AbstractSparseArray})
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end
    first_iter = true
    gradient = 0
    while t <= max_iteration && dual_gap >= max(epsilon, eps())

        #####################
        # managing time and Ctrl-C
        #####################
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

        #####################
        batch_size = batchsize_iterate(batch_iterator)

        if momentum_iterator === nothing
            gradient = compute_gradient(
                f,
                x,
                rng=rng,
                batch_size=batch_size,
                full_evaluation=full_evaluation,
            )
        elseif first_iter
            gradient = copy(compute_gradient(
                f,
                x,
                rng=rng,
                batch_size=batch_size,
                full_evaluation=full_evaluation,
            ))
        else
            momentum = momentum_iterate(momentum_iterator)
            compute_gradient(
                f,
                x,
                rng=rng,
                batch_size=batch_size,
                full_evaluation=full_evaluation,
            )
            # gradient = momentum * gradient + (1 - momentum) * f.storage
            LinearAlgebra.mul!(gradient, LinearAlgebra.I, f.storage, 1-momentum, momentum)
        end
        first_iter = false

        v = compute_extreme_point(lmo, gradient)

        # go easy on the memory - only compute if really needed
        if (mod(t, print_iter) == 0 && verbose) ||
           callback !== nothing ||
           !(line_search isa Agnostic || line_search isa Nonconvex || line_search isa FixedStep)
            primal = compute_value(f, x, full_evaluation=true)
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        end

        if line_search isa Agnostic
            gamma = 2 // (2 + t)
        elseif line_search isa Nonconvex
            gamma = 1 / sqrt(t + 1)
        elseif line_search isa Shortstep
            gamma = dual_gap / (L * norm(x - v)^2)
        elseif line_search isa RationalShortstep
            rat_dual_gap = sum((x - v) .* gradient)
            gamma = rat_dual_gap // (L * sum((x - v) .^ 2))
        elseif line_search isa FixedStep
            gamma = gamma0
        end

        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap,
                dual_gap=dual_gap,
                time=tot_time,
                x=x,
                v=v,
                gamma=gamma,
            )
            callback(state)
        end

        @emphasis(emphasis, x = (1 - gamma) * x + gamma * v)

        if mod(t, print_iter) == 0 && verbose
            tt = regular
            if t == 0
                tt = initial
            end
            rep = (
                st[Symbol(tt)],
                string(t),
                Float64(primal),
                Float64(primal - dual_gap),
                Float64(dual_gap),
                tot_time,
                t / tot_time,
                batch_size,
            )
            print_callback(rep, format_string)
            flush(stdout)
        end
        t += 1
    end
    # recompute everything once for final verfication / no additional callback call
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # last computation done with full evaluation for exact gradient

    (primal, gradient) = compute_value_gradient(f, x, full_evaluation=true)
    v = compute_extreme_point(lmo, gradient)
    # @show (gradient, primal)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = last
        tot_time = (time_ns() - time_start) / 1.0e9
        rep = (
            st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            tot_time,
            t / tot_time,
            batch_size
        )
        print_callback(rep, format_string)
        print_callback(nothing, format_string, print_footer=true)
        flush(stdout)
    end
    return x, v, primal, dual_gap, traj_data
end
