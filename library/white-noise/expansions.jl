using ForwardDiff
using SpecialFunctions

∂(f,x) = ForwardDiff.derivative(f,x)
∂²(f,x) = ∂(x -> ∂(f,x),x)

function ρ_differentiable(ρ)
    abs(∂(ρ,-eps())) < 1e-10
end


########################################################
## SINGLE MODE
########################################################

    ## Domain interior
    function construct_whitenoise_fitness_perturbation__int_diff(σ,ρ,ε̃::Number)
        c₀ = ε̃ - σ(ε̃)^2 / 2
        c₁ = 0.0
        c₂ = -σ(ε̃) / (2π) * ((π - 2) * σ(ε̃) * ∂²(ρ,0.0) + π * ∂²(σ,ε̃))
        [c₀,c₁,c₂]
    end
    function construct_whitenoise_fitness_perturbation__int_notdiff(σ,ρ,ε̃::Number)
        c₀ = ε̃ - σ(ε̃)^2 / 2
        c₁ = sqrt(2) * σ(ε̃)^2 * abs(∂(ρ,eps())) * (sqrt(π) - gamma(3/4)^2) / π 
        c₂ = 0.0
        [c₀,c₁,c₂]
    end
    function construct_whitenoise_fitness_perturbation__int(σ,ρ,ε̃::Number)
        if ρ_differentiable(ρ)
            return construct_whitenoise_fitness_perturbation__int_diff(σ,ρ,ε̃)
        else
            return construct_whitenoise_fitness_perturbation__int_notdiff(σ,ρ,ε̃)
        end
    end
    construct_whitenoise_fitness_perturbation__int(σ,ρ,p) = 
        construct_whitenoise_fitness_perturbation__int(ε -> σ(ε,p), ε -> ρ(ε,p),
            solve_whitenoise_quadratic_optimum(p))

    ## Domain exterior
    function construct_whitenoise_fitness_perturbation__ext_diff(σ,ρ,ε̃::Number)
        c₀ = ε̃ - σ(ε̃)^2 / 2
        c₁ = sqrt(2 / π) * σ(ε̃) * ∂(σ,ε̃) - sqrt(2 / π)
        c₂ = 0.0
        [c₀,c₁,c₂]
    end
    function construct_whitenoise_fitness_perturbation__ext_notdiff(σ,ρ,ε̃::Number)
        c₀ = ε̃ - σ(ε̃)^2 / 2
        c₁ = sqrt(2) / π * σ(ε̃) * (-abs(∂(ρ,-eps())) * gamma(3/4)^2 * σ(ε̃) + sqrt(π) * (σ(ε̃) * ∂(ρ,-eps()) + ∂(σ,ε̃))) - sqrt(2 / π)
        c₂ = 0.0
        [c₀,c₁,c₂]
    end
    function construct_whitenoise_fitness_perturbation__ext(σ,ρ,ε̃::Number)
        if ρ_differentiable(ρ)
            return construct_whitenoise_fitness_perturbation__ext_diff(σ,ρ,ε̃)
        else
            return construct_whitenoise_fitness_perturbation__ext_notdiff(σ,ρ,ε̃)
        end
    end
    construct_whitenoise_fitness_perturbation__ext(σ,ρ,p::Vector) = 
        construct_whitenoise_fitness_perturbation__ext(ε -> σ(ε,p), ε -> ρ(ε,p),
            solve_whitenoise_quadratic_optimum(p))


"""
    Construct perturbation expansion for the two-source white-noise model with
    heterogeneity around a single mode.

    Returns a vector [c₀,c₁,c₂] such that the fitness is given by
    f(η) ∼ c₀ + c₁ η + c₂ η²
"""
function construct_whitenoise_fitness_perturbation(σ,ρ,ε̃::Number)
    if 0.0 < ε̃ < 1.0
        return construct_whitenoise_fitness_perturbation__int(σ,ρ,ε̃)
    else
        return construct_whitenoise_fitness_perturbation__ext(σ,ρ,ε̃)
    end
end
construct_whitenoise_fitness_perturbation(σ,ρ,p::Vector) = 
    construct_whitenoise_fitness_perturbation(ε -> σ(ε,p), ε -> ρ(ε,p),
    solve_whitenoise_quadratic_optimum(p))


########################################################
## BINARY MODEL
########################################################

function construct_binary_whitenoise_fitness_perturbation(μ,σ,ρ,q)
    
    c₀ = μ(1) - 1/2 * q * (-2μ(0) + 2μ(1) + q * σ(0)^2) - 
            (1 - q) * q * ρ(1) * σ(0) *  σ(1) - 
            1/2 * (-1 + q)^2 * σ(1)^2

    c₁ = sqrt(2 / π) * q * (
        ∂(μ,0) + (-1 + q) * ρ(1) * σ(1) * ∂(σ,0) + 
        σ(0) * ((-1 + q) * σ(1) * abs(∂(ρ,1)) - q * ∂(σ,0)))

    c₂ = sqrt(2 / π) * (1 - q) * (
        -∂(μ,1) + (q * ρ(1) * σ(0) + σ(1) - 
        q * σ(1)) * ∂(σ,1) - 
        q  * sqrt(1 - ρ(1)^2) * σ(0) * σ(1) * sqrt(abs(∂²(ρ,0))))
    
    [c₀,c₁,c₂]
end