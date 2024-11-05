using Plots, StatsPlots
using Random
using Optim

include("defaults.jl")
include("../library/library.jl")

########################################################
## Setup
########################################################

    # Parameter set(s)
    p = [[√2 ,1.0,5.0],
         [1.0,1.0,5.0]]

    # Quadratic form of noise function
    σ = (ε,p) -> sqrt(p[1]^2 * (p[2] * (ε^2 - ε) + ε))

    # Correlation function(s)
    ρ = [(Δ,p) -> exp(-p[3]*Δ^2), 
         (Δ,p) -> exp(-p[3]*abs(Δ))]

    # Get individual variance components (as a function of p and ρ)
   # σᵢ = p -> get_each_whitenoise_σ(σ,ρ,p) 

    # Optimal phenotype as a function of p
    ε̃ = p -> solve_whitenoise_quadratic_optimum(p)

    # Target distribution as a function of η (and p)
    p̂ = (η,p) -> Truncated(Normal(ε̃(p),η),0.0,1.0)

    # Fitness as a function of p, ρ, and η
    fitness = (η,ρ,p) -> solve_continuous_whitenoise_fitness(p̂(η,p),get_each_whitenoise_σ(σ,ρ,p))

########################################################
## Move below to library...
########################################################


########################################################
## Plot
########################################################

    # (a) - correlation functions
    fig3a = plot([Δ -> ρᵢ(Δ,p[1]) for ρᵢ in ρ],c=[:Blue :Red],label=["ρ₁" "ρ₂"],lw=2.0,
        xlabel="Δ",ylabel="Correlation",xlim=(-0.4,0.4),ylim=(0.0,1.0),widen=true)

    # (b) p[1], for each ρ
    factor = fitness(0.0,ρ[1],p[1])
    fig3b = plot([η -> fitness(η,ρᵢ,p[1]) / factor for ρᵢ in ρ],
        c=[:Blue :Red],label=["ρ₁" "ρ₂"],lw=2.0)

        # Get perturbations
        c = [construct_whitenoise_fitness_perturbation(σ,ρᵢ,p[1]) for ρᵢ in ρ]
        func = [η -> dot(cᵢ,[1.0,η,η^2]) / factor for cᵢ in c]

        # Plot perturbations
        plot!(fig3b,func,xlim=(0.0,0.1),c=:black,lw=2.0,ls=[:dash :dashdot],label="η ≪ 1")

        # Style
        plot!(fig3b,xlabel="η",ylabel="Relative fitness",xlim=(0.0,0.5),widen=true,
            ylim=(0.95,1.15),yticks=0.95:0.05:1.15)

    # (c) p[2], for each ρ
    factor = fitness(0.0,ρ[1],p[2])
    fig3c = plot([η -> fitness(η,ρᵢ,p[2]) / factor for ρᵢ in ρ],
        c=[:Blue :Red],label=["ρ₁" "ρ₂"],lw=2.0)

        # Get perturbations
        c = [construct_whitenoise_fitness_perturbation(σ,ρᵢ,p[2]) for ρᵢ in ρ]
        func = [η -> dot(cᵢ,[1.0,η,η^2]) / factor for cᵢ in c]

        # Plot perturbations
        plot!(fig3c,func,xlim=(0.0,0.1),c=:black,lw=2.0,ls=[:dash :dashdot],label="η ≪ 1")

        # Style
        plot!(fig3c,xlabel="η",ylabel="Relative fitness",xlim=(0.0,0.5),widen=true,
            ylim=(0.95,1.15),yticks=0.95:0.05:1.15)

    # Figure 3
    fig3 = plot(fig3a,fig3b,fig3c,layout=grid(1,3),size=(700,180))
    add_plot_labels!(fig3)
    savefig(fig3,"$(@__DIR__)/svg/fig3.svg")