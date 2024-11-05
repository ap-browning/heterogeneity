using Plots, StatsPlots
using JLD2
using .Threads
using Random

include("defaults.jl")
include("../library/library.jl")

#######################################
## Setup
#######################################

    # Parameters
    q = 0.1     # Proportion of persisters   at switch
    ω = 0.5     # Switching rate
    p = [q,ω]

    # Environment
    d = [create_env_normal(),create_env_gamma()]

    # θ values to look at
    Θ₁ = [0.01,1.0,10.0]        # Density plots
    Θ₂ = lrange(0.01,10.0,16)   # Fitness plots

    # Number of SDE realisations
    nsde = 50000

#######################################
## Calculations - density plots
#######################################

    # Store surfaces
    P = Array{Matrix}(undef,length(d),length(Θ₁))

    # Compute viscosity solution
    @threads for i = eachindex(Θ₁)
        for j = 1:2
            @time P[j,i],x,y,E = solve_binary_ornstein(p,d[j],Θ₁[i];β=1e-5,xmax=0.4,ny=100,nx=101)
        end
    end

    # Normalise
    Q = deepcopy(P) # Save unnormalised
    @. P /= maximum(P)

    # Get grid
    x,y = solve_binary_ornstein(p,d[1],Θ₁[1];xmax=0.4,ny=100,nx=101,ret_grid=true)

    # QSS trace
    qss_λplt = range(-4.0,4.0,200)
    qss_xplt = [binary_equilibrium_proportion(p,λ) for λ = qss_λplt]

    # Fast mean (same for each, since each distribution as the same mean)
    fast_mean = [binary_equilibrium_proportion(p,mean(d[1])),mean(d[1])]

    @save "$(@__DIR__)/jld2/fig7ab.jld2" P x y qss_λplt qss_xplt fast_mean

#######################################
## Calculations - fitness plots
#######################################

    # Storage
    fitness_pde = [similar(Θ₂),similar(Θ₂)]
    fitness_sde = [similar(Θ₂),similar(Θ₂)]
    fitness_sde_se = [similar(Θ₂),similar(Θ₂)]

    # Loop through θ's
    @time @threads for i = eachindex(Θ₂)

        # Loop through d's
        for j = 1:2

            # Recompute those that errored
            if j == 1 || i ≤ 15
                continue
            end

            # PDE
            @time fitness_pde[j][i] = solve_binary_ornstein_fitness(p,d[j],Θ₂[i])[1]

            # SDE
            @time fitness_sde[j][i],fitness_sde_se[j][i] = simulate_binary_ornstein_fitness(p,d[j],Θ₂[i];n=nsde,ret_stat=true)
        
        end

    end

    # QSS and fast
    fitness_qss = [solve_binary_ornstein_fitness(p,dᵢ,0.0) for dᵢ in d]
    fitness_fast = [solve_binary_ornstein_fitness(p,dᵢ,Inf) for dᵢ in d]

    @save "$(@__DIR__)/jld2/fig7cd.jld2" fitness_pde fitness_sde fitness_sde_se fitness_qss fitness_fast

#######################################
## Plots
#######################################

    # Density plots
    fig7ab = [plot(title="$dist") for dist = ["Normal", "Gamma"]]

        # Load data
        @load "$(@__DIR__)/jld2/fig7ab.jld2" P x y qss_λplt qss_xplt fast_mean

        # Density plot options  
        args = (
            st=:contourf,α=0.9,lw=0.0,colorbar=:none,levels=0.2:0.2:1.0,
            clim=(0.25,1.0),xlim=(0.0,0.11),ylim=(-1.5,3.0),xlabel="x",ylabel="Growth rate",
            widen=true,xticks=0:0.02:0.1
        )

        # Each distribution
        for i = eachindex(d)

            # Density plots
            plot!(fig7ab[i],x,link_fcn.(d[i],y),P[i,3]';c=clipped_cmap(0.2,0.8,cmap=palette(:Blues)),args...)
            plot!(fig7ab[i],x,link_fcn.(d[i],y),P[i,2]';c=clipped_cmap(0.2,0.8,cmap=palette(:PuRd)),args...)
            plot!(fig7ab[i],x,link_fcn.(d[i],y),P[i,1]';c=clipped_cmap(0.2,0.8,cmap=palette(:Oranges)),args...)

            # Theory
            plot!(fig7ab[i],qss_xplt,qss_λplt,c=:black,lw=2.0,ls=:dash,label="Theory")
            scatter!(fig7ab[i],[fast_mean[1]],[fast_mean[2]],c=:black,m=:diamond,label="")

        end

    # Fitness plots
    fig7cd = [plot(title="$dist") for dist = ["Normal", "Gamma"]]

        # Load
        @load "$(@__DIR__)/jld2/fig7cd.jld2" fitness_pde fitness_sde fitness_sde_se fitness_qss fitness_fast

        # Plot moment solution for normal only
        fitness_ode = θ -> solve_binary_ornstein_fitness_moments(p,d[1],θ)
        plot!(fig7cd[1],fitness_ode,lw=2.0,c=:blue,xlim=(0.01,100.0),label="Moments ODE")

        # Each distribution
        for i = eachindex(d)

            # SDE
            scatter!(fig7cd[i],Θ₂,fitness_sde[i],yerror=1.92fitness_sde_se[i],msw=1.0,lw=2.0,msc=:grey,c=:grey,label="SDE")

            # Viscosity PDE
            plot!(fig7cd[i],Θ₂,fitness_pde[i],m=:diamond,lw=2.0,c=:red,label="Viscosity PDE")

            # Fast, QSS, homogeneous
            hline!(fig7cd[i],[mean(d[i])],lw=2.0,c=:black,ls=:dash,label="Homogeneous")
            hline!(fig7cd[i],[fitness_qss[i]],lw=2.0,c=:black,ls=:dashdotdot,label="QSS")
            hline!(fig7cd[i],[fitness_fast[i]],lw=2.0,c=:black,ls=:dot,label="Fast")

        end

        plot!(fig7cd...,xaxis=:log10,link=:all,xlim=(0.01,100),ylim=(0.75,0.9))

    # Figure 7
    fig7 = plot(fig7ab...,fig7cd...,layout=grid(1,4),size=(1000,220))
    savefig("$(@__DIR__)/svg/fig7_$(randstring(5)).svg")
    fig7