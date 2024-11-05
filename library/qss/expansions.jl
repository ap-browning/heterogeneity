using ForwardDiff
using Distributions
using QuadGK

########################################################
## EXPANSIONS FOR BINARY/EDGE MODEL
########################################################

"""
    For a fixed value of the environment, with growth rates given by
    λ = λ(ε).

    The target distribution is
        q * TN(0,η₁) + (1 - q) * TN(1,η₂)
    and the fitness expansion is of the form
        F ~ c₀ + c₁ * η₁ + c₂ * η₂
    where 
        c = [c₀,c₁,c₂]
    are returned (unless ret_fun = true, in which case f(η₁,η₂) is returned.)

"""
function construct_fitness_binary_perturbation_fixedenv(p,λ::Function)

    # Get parameters...
    q,ω = p

    # Derivative of growth rate within the phenotypic space...
    λ′(ε) = ForwardDiff.derivative(ε -> λ(ε),ε)

    # Intermediate parameters
    Δλ = λ(1) - λ(0)
    Δ = sqrt((ω - Δλ)^2 + 4*(1 - q) * ω * Δλ)

    # Initial proportion
    γ₀ = binary_equilibrium_proportion(p,Δλ)

    # Coefficients
    c₀ = γ₀ * λ(0) + (1 - γ₀) * λ(1)
    c₁ = λ′(0) / sqrt(2π) * (1 + ((2q - 1) * ω - Δλ) / Δ)
    c₂ = λ′(1) / sqrt(2π) * (-1 + ((2q - 1) * ω - Δλ) / Δ)
    [c₀,c₁,c₂]

end


"""
    Perturbative expansion for a full distribution.

    λ = λ(ε,z)
"""
function construct_fitness_binary_perturbation(p,λ::Function,d=Normal();q=1e-6)

    # Function to integrate
    func = z -> construct_fitness_binary_perturbation_fixedenv(p,ε -> λ(ε,z))

    # Integrate (or sum)
    𝔼(func,d;q)

end



########################################################
## EXPANSIONS WHERE MEAN p̂ IS IN THE MIDDLE OF THE PHENOTYPIC SPACE
########################################################

"""
    Construct perturbation expansion for the fitness, if the target ε̄ (default=0.5) is 
    on the domain interior.

    For a fixed environment only.
"""
function construct_fitness_perturbation_fixedenv(λ::Function,ω::Number;ε̄=0.0,ret_fun=false)

    λ′ = ε -> ForwardDiff.derivative(λ,ε)
    λ′′ = ε -> ForwardDiff.derivative(λ′,ε)

    [λ(ε̄),0.0,λ′(ε̄)^2 / ω + λ′′(ε̄) / 2]

end

"""
    Construct perturbation expansion for the fitness, if the target ε̄ (default=0.5) is
    on the domain interior.
"""
function construct_fitness_perturbation(λ::Function,ω::Number,d=Normal();ε̄=0.0,q=1e-6)

    # Function to integrate
    func = z -> construct_fitness_perturbation_fixedenv(ε -> λ(ε,z),ω;ε̄)

    # Integrate (or sum)
    c = 𝔼(func,d;q)

end

function construct_fitness_perturbation_gaussianenv(k,ξ,ω,σ)
    c₀ = k * ξ / sqrt(ξ^2 + 2σ^2)
    c₁ = 0.0
    c₂ = 4k^2 * σ^2 / (ξ * (ξ^2 + 4σ^2)^(3/2) * ω) - k * ξ / (ξ^2 + 2σ^2)^(3/2)
    [c₀,c₁,c₂]
end