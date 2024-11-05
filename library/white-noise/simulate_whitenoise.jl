"""

"""
function simulate_binary_whitenoise(p::Vector,epars::Vector,tend;x₀=rand(),ret_fun=false)

    # Cell parameters
    q,ω = p
    a,b = ω * (1 - q), ω * q

    # Environment parameters
    μ₁,μ₂,σ₁,σ₂,ρ = epars

    # Setup SDE
    function f(x,p,t)
        b - (a + b) * x + (x - 1) * x * (μ₁ - μ₂ + (x - 1) * σ₁^2 + (1 - 2x) * ρ * σ₁ * σ₂ + x * σ₂^2) 
    end
    function g(x,p,t)
        x * sqrt(
            (x - 1)^2 * σ₁^2 + 
            σ₂^2 * (sqrt(1 - ρ^2) - x*(ρ + sqrt(1 - ρ^2))^2)
        )
    end

    # Simulate
    prob = SDEProblem(f,g,x₀,(0.0,tend))
    if ret_fun 
        return solve(prob)
    else
        return solve(prob,saveat=tend).u[end]
    end

end


