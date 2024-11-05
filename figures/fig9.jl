using Plots, StatsPlots
using JLD2

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
    d = create_env_gamma()

    # Setup λ as a function of ε and z
    λ = (ε,z) -> ε * link_fcn(d,z)

    # p̂ as a function of η₁ and η₂
    p̂ = (η₁,η₂) -> construct_binary_distribution([η₁,η₂],q)

    # Setup fitness function (i.e., solving full QSS)
    fitness(η₁,η₂) = solve_qss_fitness(λ,p̂(η₁,η₂),ω)

    # Grid over which to plot (surface)
    N₁ = range(0.0,0.05,11)
    N₂ = range(0.0,0.02,10)

    # Grid over which to plot (coefficients)
    Q = lrange(0.0001,0.5,101)
    W = lrange(0.01,10.0,100)

#######################################
## Computations
#######################################

    # Compute fitness surface
    F = [fitness(η₁,η₂) for η₁ in N₁, η₂ in N₂]

    # Coefficients
    @time C = [construct_fitness_binary_perturbation([q,ω],λ) for q in Q, ω in W]

    # Save
    @save "$(@__DIR__)/jld2/fig9.jld2" N₁ N₂ F Q W C

#######################################
## Plot
#######################################

    # (a) - surface

        # Load data
        @load "$(@__DIR__)/jld2/fig9.jld2" N₁ N₂ F Q W C

        # Plot surface
        fig9a = surface(N₁,N₂,F',c=cgrad(palette(:bwr)[40:-1:1]),xaxis=:flip)
        wireframe!(fig9a,N₁,N₂,F',fillalpha=0.0,lw=1.0)
        zlim = zlims(fig9a)

        # Create perturbation
        c = construct_fitness_binary_perturbation(p,λ)

        # Plot vectors
        sc₁,sc₂ = 0.03,0.01
        plot!(fig9a,[0.0,sc₁],[0.0,0.0],[c[1],c[1] + c[2]*sc₁],lw=2.0,c=:black,label="")
        plot!(fig9a,[0.0,0.0],[0.0,sc₂],[c[1],c[1] + c[3]*sc₂],lw=2.0,c=:black,label="")
        plot!(fig9a;zlim=zlim,xlabel="η₁",ylabel="η₂",zlabel="Fitness",camera=(45,20))

    # (b) - coefficients

        # c₁
        plt1 = contourf(W,Q,getindex.(C,2),axis=:log10,clim=(-0.15,0.25),c=centered_cmap(-0.15,0.25),lw=0.0)
        contour!(plt1,W,Q,getindex.(C,1) .- mean(d),levels=[0.0],lw=2.0,ls=:dash,c=:black)
        contour!(plt1,W,Q,getindex.(C,2),levels=[0.0],lw=2.0,ls=:dashdot,c=:black)
        scatter!(plt1,[p[2]],[p[1]],c=:black,msw=0.0,shape=:diamond,label="",widen=false)

        # c₂
        plt2 = contourf(W,Q,getindex.(C,3),axis=:log10,c=palette(:bwr)[end:-1:55],levels=6,lw=0.0)
        contour!(plt2,W,Q,getindex.(C,1) .- mean(d) .- 0.55,levels=[-0.55],lw=2.0,ls=:dash,c=:black,clim=(-0.9,-0.4))
        scatter!(plt2,[p[2]],[p[1]],c=:black,msw=0.0,shape=:diamond,label="",widen=false)

        fig9b = plot(plt1,plt2,xlim=(0.01,10.0),ylim=(1e-4,:auto),yticks=10.0.^(-4.0:0.0),
            xlabel="ω",ylabel="q",size=(450,150))

    # Fig 9
    fig9 = plot(fig9a,fig9b,layout=@layout([a;b{0.2h}]),size=(450,600))
    savefig(fig9,"$(@__DIR__)/svg/fig9.svg")
    fig9