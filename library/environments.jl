#=
    
    Stationary distributions and dispatches to simulate continuously fluctuating environments.

=#

using Distributions
using Roots

include("gamma.jl")

########################################################
## WHITE NOISE ENVIRONMENT (SIMULATE ONLY)
########################################################

"""
    Create a two-phenotype white noise environment. 
"""
function create_env_whitenoise(μ::Vector,σ::Vector,ρ::Number)
    Σ = [σ[1]^2 prod(σ) * ρ; prod(σ) * ρ σ[2]^2]
    MvNormal(μ,Σ)
end

########################################################
## STATIONARY DISTRIBUTIONS
########################################################

"""
    Create Poisson environment. 
    Defaults based on growth rates in Kussel (2005). 
"""
function create_env_poisson(μ::Vector,α::Number)
    μ₁,μ₂ = μ
    Bernoulli(1 - α) * (μ₁ - μ₂) + μ₂
end
create_env_poisson() = create_env_poisson([2.0,-4.0],0.2)
create_env_poisson(α::Number) = create_env_poisson([2.0,-4.0],α)

"""
    Gamma environment. 
    Defaults to matching the mean and proportion of time spent in stress as the Poisson environment.
"""
function create_env_gamma(d=create_env_poisson();match=:stress,ω=-1.0)
    if match == :stress
        func(σ) = cdf(GammaAlt(mean(d),σ,ω),0.0) - cdf(d,0.0)
        σ = find_zero(func,(1e-5,5.0))
    else
        σ = std(d)
    end
    return GammaAlt(mean(d),σ,ω)
end


"""
    Normal environment. 
    Defaults to matching the mean and proportion of time spent in stress as the Poisson environment.
"""
create_env_normal(d=create_env_poisson();kwargs...) = create_env_gamma(d;ω=0.0,kwargs...)


########################################################
## LINK FUNCTIONS (TO STANDARD NORMAL)
########################################################

"""
    Link standard normal (z) to growth rate through an environment (d).
"""
function link_fcn(d::Distribution,z::Number)
    quantile(d,cdf(Normal(),z))
end
function link_fcn(d::Normal,z::Number)
    μ,σ = params(d)
    μ + σ * z
end


########################################################
## BINARY MODEL, STATIONARY PROPORTION
########################################################
function binary_equilibrium_proportion(p,λ::Number)
    # Cell parameters
    q,ω = p
    a,b = ω * (1 - q), ω * q
    # Proportion
    (a + b + λ - sqrt((a + b + λ)^2 - 4*b*λ)) / 2λ
end
binary_equilibrium_proportion(p,d::Distribution,z::Number) = binary_equilibrium_proportion(p,link_fcn(d,z))