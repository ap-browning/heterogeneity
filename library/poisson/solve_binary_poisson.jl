#=#################

    Solve stationary Fokker-Planck equation for the 
    binary model under the Poisson environment.

    Additionally compute the fitness.

=##################

using Roots
using LinearAlgebra
using DataInterpolations

function solve_binary_poisson(p,d,θ,β=1e-4;nx=201,x=binary_poisson_range(p,d,nx))
    ## Parameters

        # Cell parameters
        q,ω = p
        a,b = ω * (1 - q), ω * q
    
        # Environment parameters
        μ₁,μ₂ = reverse(extrema(d))
        α = 1 - d.ρ.p
        η₁,η₂ = θ / (1 - α), θ / α

    ## Within-environment advection
    f(x,λ) = b - x * (a + b + λ * (1 - x))

    ## Discretisation

        # Inner edge locations
        x̂ = x[2:end-1]

        # Centre locations
        x̄ = (x[1:end-1] + x[2:end]) / 2

        # Cell widths
        Δx = diff(x)

        # Distance between centres
        Δx̄ = (Δx[1:end-1] + Δx[2:end]) / 2

    ## Velocities
    M₁,M₂ =  [f.(x̂,μᵢ) for μᵢ in [μ₁,μ₂]]

    ## Compute Jacobian of finite-volume discretisation
    J = jac_advection(Δx,M₁,M₂) + jac_transfer(η₁,η₂,nx) + β * jac_diffusion(Δx,Δx̄)

    ## Nullspace, solution
    ℙvec = reshape(nullspace(J),nx-1,2)
    ℙvec /= sum(reshape(nullspace(J),nx-1,2) .* Δx)

    ## Warn if oscillations/instability detected
    any(ℙvec .< -1e-5) && @warn("Instability detected. Suggest increasing β.")

    ## Return...
    return ℙvec,x̄,Δx

end

"""
    Extreme values of x in the stationary distribution of x(t) under the Poisson environment.
"""
function binary_poisson_extrema(p,d)
    # Parameters
    q,ω = p
    a,b = ω * (1 - q), ω * q

    # Within-environment advection
    f(x,λ) = b - x * (a + b + λ * (1 - x))

    # Extrema
    sort([find_zero(x -> f(x,μ),(0.0,1.0)) for μ in extrema(d)])
end


"""
    Default range for binary Poisson environment.
"""
function binary_poisson_range(p,d,nx)
    # Extrema
    l,u = binary_poisson_extrema(p,d)
    m = (l + u) / 2

    # Log-space from each end
    nx1 = Int(round(nx / 2))
    [exp.(range(log(l),log(m),nx1)); 
        1 .- exp.(range(log(1 - m),log(1 - u),nx-nx1+1))[2:end]]
end

"""
    Sample from (environmentally conditional) stationary distribution
"""
function create_binary_poisson_sampler(P,x,Δ)
    # Get extrema
    l = x[1] - Δ[1] / 2
    u = x[end] + Δ[end] / 2
    # Construct interpolation for rejection sampling
    itp = [LinearInterpolation(P[:,i] ./ maximum(P[:,i]),x,extrapolate=true) for i = 1:2]
    # Propose until criteria reached
    propose() = rand() * (u - l) + l
    function sample(i)
        xᵢ = propose()
        while itp[i](xᵢ) .< rand()
            xᵢ = propose()
        end
        return xᵢ
    end
    return sample
end

########################################################
## FINITE VOLUME DISCRETISTAION
########################################################

function jac_advection(Δx,M₁,M₂)
    # For reuse
    Δx₁ = Δx[2:end]   ./ (Δx[1:end-1] + Δx[2:end])
    Δx₂ = Δx[1:end-1] ./ (Δx[1:end-1] + Δx[2:end])
    # State 1
    u₁ = -M₁ .* Δx₁ ./ Δx[1:end-1]
    l₁ = M₁ .* Δx₂ ./ Δx[2:end]
    m₁ = ([-M₁ .* Δx₂;0] + [0;M₁ .* Δx₁]) ./ Δx
    # State 2
    u₂ = -M₂ .* Δx₁ ./ Δx[1:end-1]
    l₂ = M₂ .* Δx₂ ./ Δx[2:end]
    m₂ = ([-M₂ .* Δx₂;0] + [0;M₂ .* Δx₁]) ./ Δx
    Tridiagonal([l₁; 0.0; l₂],[m₁; m₂],[u₁; 0.0; u₂])
end
function jac_diffusion(Δx,Δx̄)
    # Upper diagonal
    û = 1 ./ (Δx[1:end-1] .* Δx̄)
    u = [û; 0.0; û]
    # Lower diagonal
    l̂ = 1 ./ (Δx[2:end] .* Δx̄)
    l = [l̂; 0.0; l̂]
    # Main diagonal (conservative)
    nx = length(Δx) + 1
    m = zeros(2nx - 2)
    m[1:end-1] -= u
    m[2:end]   -= l
    Tridiagonal(l,m,u)
end
function jac_transfer(η₁,η₂,nx)
    B₁₁ = -η₁ * I(nx-1)
    B₂₂ = -η₂ * I(nx-1)
    [B₁₁ -B₂₂; -B₁₁ B₂₂]
end



########################################################
## FITNESS
########################################################

"""
    Fitness (general)
"""
function solve_binary_poisson_fitness(d::Distribution,P,x,Δ)
    # Environment parameters
    μ₁,μ₂ = reverse(extrema(d))
    # Calculate fitness
    sum(P .* [μ₁ μ₂] .* (1 .- x) .* Δ)
end

"""
    Fitness (QSS)
"""
function solve_binary_poisson_fitness_qss(p,d)
    # Environment parameters
    μ₁,μ₂ = reverse(extrema(d))
    α = 1 - d.ρ.p
    # Stationary environment
    l,u = binary_poisson_extrema(p,d)
    # QSS fitness
    (1 - α) * (1 - l) * μ₁ + α * (1 - u) * μ₂
end

"""
    Fitness (fast)
"""
function solve_binary_poisson_fitness_fast(p,d)
    μ̄ = mean(d)
    x̄ = binary_equilibrium_proportion(p,μ̄)
    (1 - x̄) * μ̄
end

""" 
    Fitness (general)
"""
function solve_binary_poisson_fitness(p::Vector,d,θ;nx=201,β=1e-4)
    # QSS limit
    θ == 0.0 && return solve_binary_poisson_fitness_qss(p,d)
    # Fast limit
    isinf(θ) && return solve_binary_poisson_fitness_fast(p,d)
    # General
    P,x,Δ = solve_binary_poisson(p,d,θ,β;nx)
    solve_binary_poisson_fitness(d,P,x,Δ)
end
