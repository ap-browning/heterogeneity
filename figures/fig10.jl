using Plots, StatsPlots
using JLD2

include("defaults.jl")
include("../library/library.jl")

#######################################
## Setup
#######################################

    # Switching rate
    ω = 1.5

    # Environment
    σ = 0.1
    d = Normal(0.5,σ)

    # Growth rate as a function of phenotype and environment
    ξ = 0.05
    k = 1.0
    λ = (ε,z) -> k * exp(-(ε - z)^2 / ξ^2)

    # Single fittest phenotype
    ε̄ = mean(d)

    # Target distribution
    p̂ = η -> Normal(ε̄,η)

    # Values to plot
    N = range(0.0,0.1,101)

    # Fitness function
    fitness = η -> solve_qss_fitness(λ,p̂(η),ω,d;λrange=(0.0,k))

#######################################
## Computations
#######################################

    # Calculate fitness
    @time F = fitness.(N)

    # Save results
    @save "$(@__DIR__)/jld2/fig10.jld2" N F

#######################################
## Plots
#######################################

    # (a) - example environment and fitness (functions)
    fig10a = plot(xlim=(0.2,0.8),xlabel="Phenotype",yticks=0:2:10)

        # Simulate and plot an environment
        tend = 11.0; tplt = range(0.0,tend,400)
        #z = simulate_env(d,1.0,tend)
        plot!(fig10a,z.(tplt),tplt,label="Environment",lw=2.0,c=:black,α=0.2,ylim=(0.0,11.0))

        # Plot fitness function at two time points
        tpts = [9.0,10.0]; zpts = z.(tpts)
        for (i,t) in enumerate(tpts)
            scatter!(fig10a,[z(t)],[t],c=[:Red :Orange][i],msw=0.0,label="")
            plot!(fig10a,[z(t),z(t)],[t,8.0],c=[:Red :Orange][i],lw=2.0,label="t = $t")
            plot!(fig10a,[z(t),z(t)],[t,0.0],c=[:Red :Orange][i],lw=2.0,ls=:dash,label="")
        end
        fig10a
        plot!(twinx(),[ε -> λ(ε,z) for z in zpts],c=[:Red :Orange],label="",lw=2.0,
            ylim=(0.0,1.25*1.1),xlim=(0.2,0.8),yticks=0:0.25:1.25)

        vline!(fig10a,[mean(d)],c=:black,lw=2.0,ls=:dot,label="")
        plot!(fig10a,grid=:on,box=:on)

    # (b) - fitness and perturbation expansion
    fig10b = plot(xlabel="η",ylabel="Relative fitness")

        # Load data
        @load "$(@__DIR__)/jld2/fig10.jld2" N F

        # Relative-ise the fitness for comparison
        factor = F[1]

        # Plot numerical solution
        plot!(fig10b,N,F / factor,lw=2.0,c=:blue,label="Numerical")

        # Create and plot perturbation solution
        c = construct_fitness_perturbation_gaussianenv(k,ξ,ω,σ)
        plot!(fig10b,η -> dot(η.^(0:2),c / factor),c=:black,xlim=(0.0,0.02),lw=2.0,ls=:dash,label="η ≪ 1")
        plot!(fig10b,xlim=extrema(N),widen=true,ylim=(1.0,1.1),yticks=1.0:0.02:1.1)

    # Figure 10
    fig10 = plot(fig10a,fig10b,size=(600,220))
    add_plot_labels!(fig10)
    savefig(fig10,"$(@__DIR__)/svg/fig10.svg")
