using DifferentialEquations
using Distributions
using Plots, StatsPlots

########################################################
## SIMULATE ENVIRONMENT
########################################################

"""
    Simulate Ornstein-Uhlenbeck driven environment
"""
function simulate_env(d::ContinuousDistribution,θ,tend;dt=0.01)

    # Setup SDE
    f(x,p,t) = -θ * x
    g(x,p,t) = sqrt(2θ)

    # Solve SDE
    sol = solve(SDEProblem(f,g,randn(),(0.0,tend)),EM();dt)

    # Map to growth rate and return
    t -> link_fcn(d,sol(t))

end

"""
    Simulate a (second-order) Ornstein-Uhlenbeck driven environment
"""
function simulate_env_ou2(d::ContinuousDistribution,θ,tend;k=1,dt=0.01)

    # Setup SDE
    k₂ = -k; k₁ = sqrt(k₂^2 - k₂ * θ)
    function f!(du,u,p,t)
        du[1] = -θ * u[1]
        du[2] = k₁ * u[1] + k₂ * u[2] 
    end
    function g!(du,u,p,t)
        du[1] = sqrt(2θ)
        du[2] = 0.0
    end

    # Solve SDE
    sol = solve(SDEProblem(f!,g!,randn(2),(0.0,tend)),EM();dt)

    # Map to growth rate and return
    t -> link_fcn(d,sol(t)[2])

end

########################################################
## SIMULATE MODEL
########################################################

"""
    Setup an SDE for a coupled environment and population trajectory.
"""
function setup_binary_ornstein_sde(p,d,θ,tend,ic=binary_ornstein_default_ic(p,d))

    # Cell parameters
    q,ω = p
    a,b = ω * (1 - q), ω * q

    # Growth rate
    λ(y) = link_fcn(d,y)

    # Model
    function f!(du,u,p,t)
        x,y = u
        du[1] = b - x * (a + b + λ(y) * (1 - x))
        du[2] = -θ * y
    end
    function g!(du,u,p,t)
        du[1] = 0.0
        du[2] = sqrt(2θ)
    end
    SDEProblem(f!,g!,rand(ic),(0.0,tend))
end

"""
    Simulate a coupled environment and population trajectory.
"""
function simulate_binary_ornstein(p,d,θ,tend,ic=binary_ornstein_default_ic(p,d);dt=0.01)
    prob = setup_binary_ornstein_sde(p,d,θ,tend,ic)
    sol = solve(prob,EM();dt)
    x = t -> sol(t)[1]
    λ = t -> link_fcn(d,sol(t)[2])
    return x,λ
end

"""
    Simulate from the "stationary" distribution, taken as some far away tend.
"""
function simulate_binary_ornstein_tend(p,d,θ,tend=10/θ;ic=binary_ornstein_default_ic(p,d),n=1,dt=0.01)
    prob = setup_binary_ornstein_sde(p,d,θ,tend,ic)
    function sample()
        prob = remake(prob,u0=rand(ic))
        x,y = solve(prob,EM();dt,saveat=tend,verbose=false).u[end]
        [x,link_fcn(d,y)]
    end
    n == 1 && return sample()
    hcat([sample() for _ = 1:n]...)
end

########################################################
## FITNESS
########################################################

function simulate_binary_ornstein_fitness(p,d,θ,tend=max(5.0,10/θ);ic=binary_ornstein_default_ic(p,d),n=1,ret_stat=false,dt=0.01)
    X = simulate_binary_ornstein_tend(p,d,θ,tend;ic,n,dt)
    F = [(1 - xᵢ) * λᵢ for (xᵢ,λᵢ) in eachcol(X)]
    n == 1 && return F[1]
    if ret_stat
        return mean(F),std(F) / sqrt(length(F))
    else
        return F
    end
end

