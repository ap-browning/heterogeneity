using Plots, StatsPlots
using Random

include("defaults.jl")
include("../library/library.jl")

#######################################
## Setup
#######################################

    # Environment parameters
    Θ = [0.1,10.0]
    d = [create_env_poisson(), 
         create_env_normal(), 
         create_env_gamma()]

    tend = 20

#######################################
## Plots
#######################################

    # Poisson
    λ,tswitch = simulate_env(d[1],0.2,tend;ret_tswitch=true);
    T = simulate_env_poisson_trange(tswitch)
    fig5a = plot(T,λ.(T),c=:Blue,xlabel="Time [h]",lw=2.0,widen=true)

    # Continuous (fast)
    λ = [simulate_env(dᵢ,Θ[2],tend) for dᵢ in d[2:3]]
    fig5b = plot(λ,xlim=(0.0,tend),c=[:Red :Orange],xlabel="Time [h]",lw=2.0)

    # Continuous (slow)
    λ = [simulate_env(dᵢ,Θ[1],tend) for dᵢ in d[2:3]]
    fig5c = plot(λ,xlim=(0.0,tend),c=[:Red :Orange],xlabel="Time [h]",lw=2.0)
    
    # Continuous (slow, second-order)
    λ = [simulate_env_ou2(dᵢ,1.0,tend) for dᵢ in d[2:3]]
    fig5d = plot(λ,xlim=(0.0,tend),c=[:Red :Orange],xlabel="Time [h]",lw=2.0)

    # Marginal plots...
    xplt = range(-5.0,5.0,200)
    fplt = [pdf.(dᵢ,xplt) for dᵢ in d[2:3]]

    fig5e = plot(fplt,xplt,frange=0.0,c=[:Red :Orange],lw=2.0,fα=0.3,
        label=["Normal" "Gamma"],xlabel="Density")

    fig5 = plot(fig5a,fig5b,fig5c,fig5d,fig5e,link=:y,widen=true,ylim=(-4.0,3.0),yticks=-4:3,
        layout=@layout([grid(1,4) b{0.1w}]),size=(1000,200))
    add_plot_labels!(fig5)
    plot!(fig5,subplot=5,widen=false,xlim=(0.0,0.52))
    savefig("$(@__DIR__)/svg/fig5_$(randstring(5)).svg")
    fig5