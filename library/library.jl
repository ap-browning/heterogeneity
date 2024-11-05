include("environments.jl")
include("gamma.jl")

include("white-noise/simulate_whitenoise.jl")
include("white-noise/solve_whitenoise.jl")
include("white-noise/expansions.jl")

include("poisson/simulate_binary_poisson.jl")
include("poisson/solve_binary_poisson.jl")

include("ornstein-uhlenbeck/simulate_binary_ou.jl")
include("ornstein-uhlenbeck/solve_binary_ou.jl")

include("qss/solve_qss.jl")
include("qss/expansions.jl")

lrange(a,b,n) = exp.(range(log(a),log(b),n))

"""
    Obtain the expected value of a function, relative to distribution d.
"""
function 𝔼(func::Function,d::Distribution=Normal();q=1e-6)

    if isa(d,MixtureModel)
        sum(𝔼(func,d.components[i]) * pdf(d.prior,i) for i in support(d.prior))
    elseif isa(d,ContinuousDistribution)
        # Check for a degenerate distribution
        if var(d) == 0.0
            return func(mean(d))
        end
        # Integration limits    
        lims = quantile.(d,[q,1-q])
        # Numerically integrate
        return quadgk(z -> func(z) * pdf(d,z),lims...)[1]
    elseif isa(d,DiscreteDistribution)
        # Sum
        return sum(func(z) for z in support(d))
    end
end


"""
    Construct corresponding distribution
"""
function construct_binary_distribution(η::Vector,q::Number)
    MixtureModel(Truncated.(Normal.([0.0,1.0],η),0.0,1.0),[q,1-q])
end