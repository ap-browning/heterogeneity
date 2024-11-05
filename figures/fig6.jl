using Plots, StatsPlots
using JLD2

include("defaults.jl")
include("../library/library.jl")

#######################################
## Setup
#######################################

    # Parameters
    q = 0.1     # Proportion of persisters at switch
    ω = 1.0     # Switching rate
    p = [q,ω]

    # Fitness as a function of α and θ
    function fitness(α,θ;nx=201,β=1e-4)
        
        # Create environment
        d = create_env_poisson(α)

        # Calculate fitness
        solve_binary_poisson_fitness(p,d,θ;nx,β)

    end

    # Range of parameters to plot
    α = lrange(0.01,0.99,51)
    θ = lrange(0.001,10.0,50)

#######################################
## Calculations
#######################################

    # General regime
    F = [fitness(αᵢ,θᵢ) for αᵢ = α, θᵢ = θ]

    # Calculation in the quasi-steady-state and fast regimes
    Fqss  = [fitness(αᵢ,0.0) for αᵢ = α, θᵢ = θ]
    Ffast = [fitness(αᵢ,Inf) for αᵢ = α, θᵢ = θ]

    # Fitness of a homogeneous strategy
    H = [mean(create_env_poisson(αᵢ)) for αᵢ = α, θᵢ = θ]

    # Save
    @save "$(@__DIR__)/jld2/fig6.jld2" F Fqss Ffast H

#######################################
## Plots
#######################################

    # # Arguments (keep consistent between panels)
    # args = (
    #     clim=(-0.25,2.0),
    #     c=centered_cmap(-0.25,2.0),lw=0.0,
    #     levels=100,axis=:log10,yaxis=:flip,widen=false,
    #     xlim=(0.01,1.0),xticks=[0.01,0.1,1.0],
    #     ylim=(0.01,10.0),yticks=[0.01,0.1,1.0,10.0])

    # # Load data
    # @load "$(@__DIR__)/jld2/fig6.jld2" F Fqss Ffast H

    # # Panels
    # plt1 = contourf(α,θ,F' - H';args...)
    # plt2 = contourf(α,θ,Fqss' - H';args...)
    # plt3 = contourf(α,θ,Ffast' - H';args...)

    # # Plot thresholds
    # contour!(plt1,α,θ,F' - H';levels=[0.0],lw=2.0,c=:black,ls=:dash)
    # contour!(plt2,α,θ,Fqss' - H';levels=[0.0],lw=2.0,c=:black,ls=:dash)
    # contour!(plt3,α,θ,Ffast' - H';levels=[0.0],lw=2.0,c=:black,ls=:dash)

    # # Figure (will require additional formatting)
    # fig6 = plot(plt2,plt1,plt3,layout=@layout([a{0.15h};b;c{0.15h}]),size=(500,600))
    # savefig(fig6,"$(@__DIR__)/svg/fig6.svg")
    # fig6

#######################################
## Plots (manual colormap)
#######################################

    # Arguments
    args = (
        lw=0.0,axis=:log10,yaxis=:flip,widen=false,
        xlim=(0.01,1.0),xticks=[0.01,0.1,1.0],
        ylim=(0.01,10.0),yticks=[0.01,0.1,1.0,10.0])

    # Setup colormap and contour levels
    lvls = 10
    clim = (-0.08-eps(),3.0)
    levels_neg = range(clim[1],0,lvls)
    levels_pos = range(0,clim[2],lvls)
    colors_neg = [get(palette(:bwr,rev=true),-(x - clim[1]) / clim[1] / 2) for x in levels_neg]
    colors_pos = [get(palette(:bwr,rev=true), 0.5 + x / 2clim[2]) for x in levels_pos]
    levels = [levels_neg;levels_pos[2:end]]
    cmap = [colors_neg; colors_pos]

    # Setup and save colorbars (will not come across automatically...)
    plt1_pos = contourf(α,θ,F' - H';args...,levels=levels[1:lvls],c=cmap[1:lvls],clim=(clim[1],0.0))
    plt1_neg = contourf(α,θ,F' - H';args...,levels=levels[lvls:end],c=cmap[lvls:end],clim=(0.0,clim[2]))
    savefig(plt1_pos,"$(@__DIR__)/svg/cbar1.svg")
    savefig(plt1_neg,"$(@__DIR__)/svg/cbar2.svg")
 
    # Panels
    plt1 = contourf(α,θ,F' - H';args...,levels=levels,c=cmap,clim)
    plt2 = contourf(α,θ,Fqss' - H';args...,levels=levels,c=cmap,clim)
    plt3 = contourf(α,θ,Ffast' - H';args...,levels=levels,c=cmap,clim)

    # Plot thresholds
    contour!(plt1,α,θ,F' - H';levels=[0.0],lw=2.0,c=:black,ls=:dash)
    contour!(plt2,α,θ,Fqss' - H';levels=[0.0],lw=2.0,c=:black,ls=:dash)
    contour!(plt3,α,θ,Ffast' - H';levels=[0.0],lw=2.0,c=:black,ls=:dash)

    # Figure (will require additional formatting)
    fig6 = plot(plt2,plt1,plt3,layout=@layout([a{0.15h};b;c{0.15h}]),size=(500,600))
    savefig(fig6,"$(@__DIR__)/svg/fig6.svg")
    fig6