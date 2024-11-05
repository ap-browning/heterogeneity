using DifferentialEquations
using Distributions
using Plots, StatsPlots

########################################################
## SIMULATE ENVIRONMENT
########################################################

"""
    Simulate Poisson environment.
"""
function simulate_env(d::DiscreteDistribution,θ::Number,tend::Number;ret_tswitch=false)

    # Check for a binary environment
    length(support(d)) == 2 || error("Unsupported distribution.")

    # Extract parameters
    μ₁,μ₂ = reverse(extrema(d))
    α = 1 - d.ρ.p
    η₁,η₂ = θ / (1 - α), θ / α

    # Create distribution of switch times in one cycle (env 1, env 2)
    dist = Product([Exponential(1 / η) for η in [η₁,η₂]])

    # Good number of intervals to sample from
    n = 2Int(round(θ * tend))

    # Sample intervals
    τ = rand(dist,n)
    while sum(τ) < tend
        τ = rand(dist,n)
    end
    μ = [fill(μ₁,1,n); fill(μ₂,1,n)]

    # Reorder rows if state 2 is first
    if rand(d) == μ₂
        τ = τ[[2,1],:]
        μ = μ[[2,1],:]
    end

    # Flatten
    τ = τ[:]; μ = μ[:]

    # Durations to switch times
    tswitch = cumsum(τ)

    # Trim tswitch
    tswitch = [tswitch[tswitch .< tend]; tend + eps()]

    # Solution
    sol = t -> 0 ≤ t ≤ tend ? μ[findfirst(t .≤ tswitch)] : error("Outside of time range.")

    if ret_tswitch
        return sol,tswitch
    else
        return sol
    end

end

"""
    (Useful) get nice t-grid for plotting
"""
function simulate_env_poisson_trange(tswitch;eps=1e-5)
    sort([0.0; tswitch[1:end-1] .+ eps; tswitch .- eps])
end

########################################################
## SIMULATE MODEL
########################################################

"""
    Simulate a coupled environment and population trajectory.
"""
function simulate_binary_poisson(p,d,θ,tend;x₀ = 0.5,ret_tswitch=false)

    # Cell parameters
    q,ω = p
    a,b = ω * (1 - q), ω * q

    # Sample environment
    λ,tswitch = simulate_env(d,θ,tend;ret_tswitch=true)
    tstops = simulate_env_poisson_trange(tswitch)
    
    # Initial condition
    if isa(x₀,Function)
        ic = λ(0) == maximum(d) ? 1 : 2
    else
        ic = x₀
    end

    # ODE
    function ode(x,p,t)
        b - x * (a + b + λ(t) * (1 - x))
    end 

    x = solve(ODEProblem(ode,ic,(0.0,tend));tstops,verbose=false)

    if ret_tswitch
        return x,λ,tswitch
    else
        return x,λ
    end

end

"""
    Simulate from the "stationary" distribution, taken as some far away tend.
"""
function simulate_binary_poisson_tend(p,d::DiscreteDistribution,θ,tend;x₀=:sample,n=1)
    if x₀ == :sample
        P,x,Δ = solve_binary_poisson(p,d,θ)
        x₀ = create_binary_poisson_sampler(P,x,Δ)
    end
    function sample()
        x,λ = simulate_binary_poisson(p,d,θ,tend;x₀)
        i = λ(tend) == maximum(d) ? 1 : 2
        # Reject, try again...
        if x(tend) > 1.0
            return sample()
        end
        return [x(tend),i]
    end
    if n == 1
        return sample()
    else
        return hcat([sample() for _ = 1:n]...)
    end
end

########################################################
## TESTS
########################################################

"""
    Test the model simulator.

    using Plots, StatsPlots
    test_simulate_env_poisson()
"""
function test_simulate_binary_poisson(p,d,θ,tend;n=1000)

    # Solve for binary Poisson stationary distribution
    P,x,Δ = solve_binary_poisson(p,d,θ)

    # Setup a sampler for the SDE, sample from the SDE
    X = simulate_binary_poisson_tend(p,d,θ,tend;n)

    # Marginals
    P = [P[:,i] / sum(P[:,i] .* Δ) for i = 1:2]

    # Plot both marginals
    plt1 = plot(title="(a) State 1")
    plt2 = plot(title="(b) State 2")
    histogram!(plt1,X[1,X[2,:] .== 1.0],normalize=:pdf,lw=0.0,α=0.5,c=:blue,nbins=100)
    histogram!(plt2,X[1,X[2,:] .== 2.0],normalize=:pdf,lw=0.0,α=0.5,c=:orange,nbins=100)
    plot!(plt1,x,P[1],c=:black)
    plot!(plt2,x,P[2],c=:black)

    plot(plt1,plt2,size=(700,300))

end
test_simulate_env_poisson(n=1000) = test_simulate_binary_poisson([0.1,1.0],create_env_poisson(),0.5,20;n)

"""
    Test the environment simulator.

    using Plots, StatsPlots
    test_simulate_env_poisson()
    test_simulate_env_poisson(create_env_poisson(),0.1,50)
"""
function test_simulate_env_poisson(d,θ,tend;n=1000)

    ## (a) : Plot an environment
    plt1 = plot(title="(a) Single realisation")

    λ,tswitch = simulate_env_poisson(d,θ,tend;ret_tswitch=true)
    T = simulate_env_poisson_trange(tswitch)
    plot!(plt1,T,λ.(T))

    ## (b) : Proportion of time in each environment
    plt2 = plot(title="(b) Stationary distribution")

    λ₀ = [simulate_env_poisson(d,θ,tend)(tend / 2) for _ = 1:n]

    bar!(plt2,[extrema(d)...],[count(λ₀ .== μ) for μ in extrema(d)] / length(λ₀),bar_width=1,label="Empirical")
    hline!([d.ρ.p],lw=2.0,c=:black,ls=:dash,label="Theory")

    ## (c) : Cycle duration
    plt3 = plot(title="(c) Cycle duration")

    τ₁ = [simulate_env_poisson(d,θ,10tend;ret_tswitch=true)[2][2] for _ = 1:n]
    density!(plt3,τ₁,lw=2.0,c=:blue,label="Cycle 1 (empircal)")
    vline!([mean(τ₁)],lw=2.0,c=:blue,ls=:dash,label="")
    vline!([1 / θ],lw=2.0,c=:black,ls=:dash,label="1/θ")

    ## Plot
    plot(plt1,plt2,plt3,layout=grid(1,3),size=(900,300))

end
test_simulate_env_poisson() = test_simulate_env_poisson(create_env_poisson(),0.1,50)