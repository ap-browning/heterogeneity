using Plots, StatsPlots
using Random
using Optim

include("defaults.jl")
include("../library/library.jl")

########################################################
## Setup
########################################################

    # Environmental parameters
    μ = [1.0,0.3]
    σ = [1.0,0.1]
    ρ = -0.5
    epars = [μ;σ;ρ]

    # Cell parameters (optimal switching rate)
    q = solve_binary_whitenoise_optimalq(epars)
    ω = 1.0
    p = [q,ω]

    # tend
    tend = 50.0

########################################################
## (a) Two-phenotypes in a white noise environment
########################################################

    # Environmental parameters
    μ = [1.0,0.3]
    σ = [1.0,0.1]
    ρ = -0.5

    d = create_env_whitenoise(μ,σ,ρ)

    # Simulate
    tplt = range(-1.0,11.0,500)
    fig2a = plot(tplt,rand(d,length(tplt))',c=[:Blue :Red],lw=2.0,
        label=["Phenotype 1 (ε = 1)" "Phenotype 2 (ε = 0)"])
    plot!(fig2a,ylim=(-2.0,4.0),xticks=0:2:10,xlim=(0.0,10.0),widen=true)
    plot!(fig2a,xlabel="Time [h]",ylabel="Growth rate")


########################################################
## (b) Stationary distribution (for finite switching rate)
########################################################

    # Simulated stationary distribution
    @time X = [simulate_binary_whitenoise(p,epars,tend) for _ = 1:10000];

    # Solution (from PDE)
    @time g = solve_binary_whitenoise(p,epars)

    # Plot
    fig2b = plot(xlabel="x(t)",ylabel="Density") 
    plot!(fig2b,g,c=:blue,lw=2.0,frange=0.0,fα=0.2,label="PDE")
    plot!(fig2b,X,st=:density,c=:orange,lw=2.0,label="SDE")
    vline!([q],c=:black,lw=2.0,ls=:dash,label="q̂")

########################################################
## (c) Optimal q for finite ω
########################################################

    # Range of ω to plot
    W = [0.1,0.5,1.0,2.0,10.0]

    # Fitness as a function of q and ω
    fitness = (q,ω) -> solve_binary_whitenoise_fitness([q,ω],epars)

    # Get optimal q for each ω
    optim_q = ω -> solve_binary_whitenoise_optimalq(ω,epars)
    optim_q.(W)

    # Plot for each W
    funcs = [q -> fitness(q,ω) for ω in W]
    fig2c = plot()
    for (i,ω) = enumerate(W)
        q̂,f̂ = optim_q(ω)
        plot!(fig2c,funcs[i],c=palette(:Blues_7)[2+i],lw=2.0,label="ω = $ω")
        scatter!(fig2c,[q̂],[f̂],c=palette(:Blues_7)[2+i],label="")
    end
    vline!([q],c=:black,lw=2.0,ls=:dash,label="q̂")
    plot!(fig2c,ylim=(0.495,0.565),widen=true,xlabel="q",ylabel="Fitness")

########################################################
## Figure 2
########################################################

fig2 = plot(fig2a,fig2b,fig2c,layout=grid(1,3),size=(700,180))
add_plot_labels!(fig2)
savefig(fig2,"$(@__DIR__)/svg/fig2.svg")