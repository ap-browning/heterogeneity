using QuadGK
using Optim

########################################################
## BINARY MODEL
########################################################

"""
    Solution to the Fokker-Planck equation for x(t).
"""
function solve_binary_whitenoise(p,epars)

    # Cell parameters
    q,ω = p
    a,b = ω * (1 - q), ω * q

    # Environment parameters
    μ₁,μ₂,σ₁,σ₂,ρ = epars

    # Unnormalised PDF
    α = x -> -2 * ((a - b + μ₁ - μ₂ - ρ * σ₁ * σ₂ + σ₂^2) * log(x) + 
             (b - a - μ₁ + μ₂ - ρ * σ₁ * σ₂ + σ₁^2) * log(1 - x) + 
              b / x - a / (x - 1)) / 
              (σ₁^2 - 2ρ*σ₁*σ₂+σ₂^2)
    g = x -> exp(α(x))

    # Normalising factor
    C = quadgk(g,eps(),1.0 - eps())[1]

    # Normalised PDF
    x -> 0 < x < 1.0 ? g(x) / C : 0.0

end


function solve_binary_whitenoise_fitness(p,epars)

    # Cell parameters
    q,ω = p
    a,b = ω * (1 - q), ω * q

    # Environment parameters
    μ₁,μ₂,σ₁,σ₂,ρ = epars

    # Get distribution of x
    f = solve_binary_whitenoise(p,epars)

    # Fitness as a function of x
    fitness = x -> μ₂ - 0.5 * (1 - x) * (-2μ₁ + 2μ₂ + (1 - x) * σ₁^2) - ρ * x * (1 - x) * σ₁ * σ₂ - 0.5 * x^2 * σ₂^2

    # Integrate
    quadgk(x -> f(x) * fitness(x),eps(),1 - eps())[1]

end

function solve_binary_whitenoise_optimalq(ω,epars)
    
    # Function to maximise
    fitness = q -> -solve_binary_whitenoise_fitness([q,ω],epars)

    # Optimize
    res = optimize(fitness,0.0,1.0)

    # Return
    (res.minimizer,-res.minimum)

end

"""
    In the ω → ∞ limit
"""
function solve_binary_whitenoise_optimalq(epars)

    # Environment parameters
    μ₁,μ₂,σ₁,σ₂,ρ = epars

    # Optimum (unbounded)
    q = 1 - (1 - μ₂ - ρ * σ₁ * σ₂ + σ₂^2) / (σ₁^2 - 2 * ρ * σ₁ * σ₂ + σ₂^2)

    # Optimum (bounded)
    return max(min(q,1.0),0.0)

end
function solve_binary_whitenoise_optimalq(m::Function,σ::Function,ρ::Function;ε=[1.0,0.0])
    # Environmental parameters
    epars = [m.(ε); σ.(ε); ρ(ε[2] - ε[1])]
    solve_binary_whitenoise_optimalq(epars)
end


########################################################
## CONTINUOUS MODEL (SINGLE MODE)
########################################################

"""
    Solve continuous white noise model using numerical integration.
"""
function solve_continuous_whitenoise_fitness(p::Distribution,m::Function,σᵢ::Function;q=1e-6)

    𝔼σᵢ = 𝔼(σᵢ,p;q)

    𝔼(m,p;q) - sum(𝔼σᵢ.^2) / 2

end
function solve_continuous_whitenoise_fitness(p::Distribution,σᵢ::Function;q=1e-6)
    m = ε -> ε
    solve_continuous_whitenoise_fitness(p,m,σᵢ;q)
end
function solve_whitenoise_fitness(q,m,σᵢ;quant=1e-6)
    p = construct_binary_distribution([0.0,0.0],q)
    solve_continuous_whitenoise_fitness(p,m,σᵢ;q=quant)
end


"""
    Get each component (as a function of phenotype) for a two-source whitenoise model.
"""
function get_each_whitenoise_σ(σ,ρ,ε̂::Number)
    σ₁ = ε -> σ(ε) * ρ(ε - ε̂)
    σ₂ = ε -> σ(ε) * sqrt(1 - ρ(ε - ε̂)^2)
    ε -> [σ₁(ε),σ₂(ε)]
end
get_each_whitenoise_σ(σ,ρ,p::Vector,ε̂::Number=solve_whitenoise_quadratic_optimum(p)) = 
    get_each_whitenoise_σ(ε -> σ(ε,p),Δ -> ρ(Δ,p),ε̂)

"""
    Get optimal single phenotype if variance function is quadratic in form.
"""
function solve_whitenoise_quadratic_optimum(p)
    max(min(1 / 2 + 1 / p[2] * (1 / p[1]^2 - 1 / 2),1.0),0.0)
end