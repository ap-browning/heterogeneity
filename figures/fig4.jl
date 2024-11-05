using Plots, StatsPlots
using Random
using Optim

include("defaults.jl")
include("../library/library.jl")

########################################################
## Setup
########################################################

    # Environmental parameters
    μ₀,μ₁ = 0.3,1.0
    σ₀,σ₁ = 0.1,1.0
    ρ₁,α = -0.5,10.0
    k = 0.3

    # epars (for discrete binary)
    epars = [μ₁,μ₀,σ₁,σ₀,ρ₁]

    # Mean (linear)
    m = ε -> μ₀ + (μ₁ - μ₀) * ε

    # Quadratic form of noise function
    σ = ε -> sqrt((σ₁^2 - σ₀^2) * (k * (ε^2 - ε) + ε) + σ₀^2)

    # Correlation function(s)
    ρ = Δ -> exp(-α*Δ^2) * (1 - ρ₁) + ρ₁

    # Optimal target distribution
    q = solve_binary_whitenoise_optimalq(m,σ,ρ)

    # Get individual variance components (as a function of p and ρ)
    σᵢ = get_each_whitenoise_σ(σ,ρ,1.0)

    # Target distribution as a function of η (and p)
    p̂ = (η₁,η₂) -> construct_binary_distribution([η₁,η₂],q)

    # Fitness as a function of p, ρ, and η
    fitness = (η₁,η₂) -> solve_continuous_whitenoise_fitness(p̂(η₁,η₂),m,σᵢ)

    # Values of each to plot
    N₁ = range(0.0,0.5,21)
    N₂ = range(0.0,0.5,21)

    # Computation
    @time F = [fitness(η₁,η₂) for η₁ in N₁, η₂ in N₂]

    # Fitness of the homogeneous strategy
    F₀ = m(1.0) - σ(1.0)^2 / 2

########################################################
## Plot
########################################################

    # Intermediate distributions
    η = [[0.4,0.4],[0.1,0.1]]
    func1 = [x -> pdf(p̂(ηᵢ...),x) for ηᵢ in η]
    func2 = [x -> fun(x) / max(fun(0.0),fun(1.0)) for fun in func1]

    fig4a = plot(func1,xlim=(0.0,1.0),frange=0.0,fα=0.2,c=[:Red :Grey],lw=2.0,
        label=["(η₁,η₂) = $ηᵢ" for  _ = 1:1, ηᵢ in η],xlabel="Phenotype",ylabel="Density")

    # Fitness surface
    fig4b = surface(N₁,N₂,F' ./ F₀,c=cgrad(palette(:bwr)[40:-1:1]),xaxis=:flip)
    wireframe!(fig4b,N₁,N₂,F' ./ F₀,fillalpha=0.0,lw=1.0)
    zlim = zlims(fig4b)

    # Construct perturbation
    c = construct_binary_whitenoise_fitness_perturbation(m,σ,ρ,q)

    # Plot vectors
    sc₁,sc₂ = 0.1,0.1
    plot!(fig4b,[0.0,sc₁],[0.0,0.0],[c[1],c[1] + c[2]*sc₁] / F₀,lw=2.0,c=:black,label="")
    plot!(fig4b,[0.0,0.0],[0.0,sc₂],[c[1],c[1] + c[3]*sc₂] / F₀,lw=2.0,c=:black,label="")
    plot!(fig4b;zlim=zlim,xlabel="η₁",ylabel="η₂",zlabel="Fitness",camera=(70,25))

    # Plot points from (a)
    for i = eachindex(η)
        scatter!([η[i][1]],[η[i][2]],[fitness(η[i]...) / F₀],c=[:Red :Grey][i],label="")
    end

    # Figure 4
    fig4 = plot(fig4a,fig4b,layout=@layout([a; b{0.7h}]))

    # Save each component seperately...
    plot!(fig4a,size=(350,150))
    #savefig(fig4a,"$(@__DIR__)/svg/fig4a.svg")

    plot!(fig4b,size=(450,450))
    #savefig(fig4b,"$(@__DIR__)/svg/fig4b.svg")