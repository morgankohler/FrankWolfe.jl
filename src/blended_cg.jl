
"""
    simplex_gradient_descent(active_set::ActiveSet, direction, f)

Performs a Simplex Gradient Descent step and modifies `active_set`.

Algorithm reference and notation taken from:
Blended Conditional Gradients:The Unconditioning of Conditional Gradients
http://proceedings.mlr.press/v97/braun19a/braun19a.pdf
"""
function update_simplex_gradient_descent!(active_set::ActiveSet, direction, f, gradient_dir ; L=nothing, linesearch_tol=10e-7, step_lim=20)
    c = [dot(direction, a) for a in active_set]
    k = length(active_set)
    csum = sum(c)
    c .-= (csum / k)
    # name change to stay consistent with the paper
    d = c
    if norm(d) <= 1e-7
        # resetting active set to singleton
        a0 = active_set.atoms
        empty!(active_set)
        push!(active_set, (1, a0))
        return active_set
    end
    η = eltype(d)(Inf)
    remove_idx = -1
    @inbounds for idx in eachindex(d)
        if d[idx] ≥ 0
            η = min(
                η,
                active_set.weights[idx] / d[idx]
            )
        end
    end
    η = max(0, η)
    x = compute_active_set_iterate(active_set)
    @. active_set.weights -= η * d
    if !active_set_validate(active_set)
        error("""
        Eta computation error?
        $η\n$d\nactive_set.weights
        """)
    end
    y = compute_active_set_iterate(active_set)
    if f(x) ≥ f(y)
        active_set_cleanup!(active_set)
        @assert(length(active_set) == length(x) - 1)
        return active_set
    end
    # TODO move η between x and y till opt
    linesearch_method = L === nothing || !isfinite(L) ? backtracking : shortstep
    if linesearch_method == backtracking
        _, gamma = backtrackingLS(f, gradient_dir, x, v, linesearch_tol=linesearch_tol, step_lim=step_lim)
    else # == shortstep, just two methods here for now
        @assert dot(gradient_dir, x - y) ≥ 0
        gamma = dot(gradient_dir, x - y) / (L * norm(x - y)^2)
    end
    @assert 0 ≤ gamma ≤ 1
    # step back from y to x by γ η d
    # new point is x - (1 - γ) η d
    @. active_set.weights += η * gamma * d
    # could be required in some cases?
    active_set_cleanup!(active_set)
    return active_set
end
