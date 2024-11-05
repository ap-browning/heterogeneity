using Plots, StatsPlots

include("defaults.jl")
include("../library/library.jl")

#######################################
## Setup
#######################################

    # QSS
    θ = 0.0

    # Environment(s)
    d = create_env_gamma()

    # Plotting range
    Q = lrange(0.0001,0.5,101)
    W = lrange(0.01,10.0,100)

#######################################
## Calculations
#######################################

    # Fitness as a function of heterogeneity strategy
    fitness(q,ω) = solve_binary_ornstein_fitness([q,ω],d,θ)

    # Calculate surface
    @time F = [fitness(qᵢ,ωᵢ) for qᵢ = Q, ωᵢ = W]

    # Fitness relative to homogeneous strategy (%)
    A = F ./ mean(d)

#######################################
## Plots
#######################################

    fig8 = contourf(W,Q,A,axis=:log,clim=(0.5,1.3),c=centered_cmap(-0.5,0.3),
        lw=0.0,xlabel="ω",ylabel="q",yaxis=:reverse,xlim=(0.01,10.0),ylim=(1e-4,0.5),
        xticks=10.0.^(-2.0:1.0),yticks=10.0.^(-4:0),size=(300,260))

    # Add region seperator
    contour!(W,Q,A,levels=[1.0],c=:black,ls=:dash)

    # Save
    savefig(fig8,"$(@__DIR__)/svg/fig8.svg")