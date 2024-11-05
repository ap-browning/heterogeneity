#= #################

    Solve the Fokker-Planck equation for the 
    binary model under continuously fluctuating environments
    driven by Wiener noise.

    Additionally compute the fitness.

=# ##################

using DifferentialEquations
using Distributions
using ForwardDiff
using QuadGK
using MomentClosure
using ModelingToolkit
using TerminalLoggers
using Krylov
using SparseArrays

########################################################
## PDE APPROACH (ALL DISTRIBUTIONS)
########################################################

"""
    Solve binary model driven by Ornstein-Uhlenbeck processes using a finite volume method.
"""
function solve_binary_ornstein(p,d,θ,tend,ic=binary_ornstein_default_ic(p,d);
    nx=101,ny=100,β=1e-5,verbose=false,ret_finaldu=false,
    x = binary_ornstein_default_xgrid(p,d,nx),
    y = range(-4.0,4.0,ny))

    ## Parameters and model

        # Cell parameters
        q,ω = p
        a,b = ω * (1 - q), ω * q

        # Growth rate
        λ(y) = link_fcn(d,y)


        # Drift terms
        fx(x,y) = b - x * (a + b + λ(y) * (1 - x))
        fy(x,y) = -θ * y

    ## Discretisation

        # Inner edge locations
        x̂ = x[2:end-1]
        ŷ = y[2:end-1]

        # Centre locations
        x̄ = (x[1:end-1] + x[2:end]) / 2
        ȳ = (y[1:end-1] + y[2:end]) / 2

        # Cell widths, and volumes
        Δx = diff(x)
        Δy = diff(y)
        dA = Δx * Δy'

        # Distance between centres
        Δx̄ = (Δx[1:end-1] + Δx[2:end]) / 2
        Δȳ = (Δy[1:end-1] + Δy[2:end]) / 2

    ## Initial condition

        if isa(ic,Distribution)
            ū = [pdf(ic,[xᵢ,yᵢ]) for xᵢ in x̄, yᵢ in ȳ]
        else
            ū = [ic(xᵢ,yᵢ) for xᵢ in x̄, yᵢ in ȳ]
        end
        ū /= sum(ū .* dA)

    ## PDE

        # Fixed quantities
        Mx = [fx(xᵢ,yᵢ) for xᵢ in x̂, yᵢ in ȳ]
        My = [fy(xᵢ,yᵢ) for xᵢ in x̄, yᵢ in ŷ]

        # Discretised PDE
        function pde!(dū,ū,p,t)
            verbose && display(t)
            ū = max.(ū,0.0)
    
            # X edges (geometric mean)
            ûx = log.(ū[1:end-1,:]) .* Δx[1:end-1] + log.(ū[2:end,:]) .* Δx[2:end]
            ûx ./= Δx[1:end-1] + Δx[2:end]
            ûx = exp.(ûx)
    
            # Y edges (geometric mean)
            ûy = log.(ū[:,1:end-1]) .* Δy[1:end-1]' + log.(ū[:,2:end]) .* Δy[2:end]'
            ûy ./= Δy[1:end-1]' + Δy[2:end]'
            ûy = exp.(ûy)
    
            # Transport flux in each direction
            f̂x = Mx .* ûx
            f̂y = My .* ûy
    
            # Viscosity in x direction
            ĝx = -β * diff(ū,dims=1) ./ Δx̄

            # Diffisive flux in each direction
            ĝy = -θ * diff(ū,dims=2) ./ Δȳ'
    
            # Total flux
            xflux = f̂x + ĝx
            yflux = f̂y + ĝy
        
            dū .= 0.0
            dū[1:end-1,:] -= xflux ./ Δx[1:end-1]
            dū[2:end,:]   += xflux ./ Δx[2:end]
            dū[:,1:end-1] -= yflux ./ Δy[1:end-1]'
            dū[:,2:end]   += yflux ./ Δy[2:end]'
    
        end

        # Solve using BS3() SSPRK22(),dt=0.001
        prob = ODEProblem(pde!,ū,(0.0,tend))
        ℙ = solve(prob,BS3(),progress=true)

        # Expectation function to return
        E = (f,t) -> sum([f(xᵢ,yᵢ) for xᵢ in x̄, yᵢ in ȳ] .* ℙ(t) .* dA)

        # Calculate final du if necessary
        if ret_finaldu
            dū = similar(ℙ(tend))
            pde!(dū,ℙ(tend),0.0,tend)
            return ℙ,x̄,ȳ,E,dū
        end

    ## Return
    return ℙ,x̄,ȳ,E

end

"""
    Stationary solution using GMRES
"""
function solve_binary_ornstein(p,d,θ;ic=binary_ornstein_default_ic(p,d),
    nx=71,ny=71,xmin=0.0005,xmax=0.5,β=1e-5,verbose=false,ret_grid=false,
    x = lrange(xmin,xmax,nx),
    y = range(-4.0,4.0,ny))

    ## Parameters and model

        # Cell parameters
        q,ω = p
        a,b = ω * (1 - q), ω * q

        # Growth rate
        λ(y) = link_fcn(d,y)

        # Drift terms
        fx(x,y) = b - x * (a + b + λ(y) * (1 - x))
        fy(x,y) = -θ * y

    ## Discretisation

        # Inner edge locations
        x̂ = x[2:end-1]
        ŷ = y[2:end-1]

        # Centre locations
        x̄ = (x[1:end-1] + x[2:end]) / 2
        ȳ = (y[1:end-1] + y[2:end]) / 2

        ret_grid && return x̄,ȳ

        # Cell widths, and volumes
        Δx = diff(x)
        Δy = diff(y)
        dA = Δx * Δy'

        # Distance between centres
        Δx̄ = (Δx[1:end-1] + Δx[2:end]) / 2
        Δȳ = (Δy[1:end-1] + Δy[2:end]) / 2

    ## Initial condition

        if isa(ic,Distribution)
            ū = [pdf(ic,[xᵢ,yᵢ]) for xᵢ in x̄, yᵢ in ȳ]
        else
            ū = [ic(xᵢ,yᵢ) for xᵢ in x̄, yᵢ in ȳ]
        end
        ū /= sum(ū .* dA)
        vecū = ū[:]

    ## PDE

        # Fixed quantities
        Mx = [fx(xᵢ,yᵢ) for xᵢ in x̂, yᵢ in ȳ]
        My = [fy(xᵢ,yᵢ) for xᵢ in x̄, yᵢ in ŷ]

        # Discretised PDE
        function pde(vecū)

            # Reshape
            ū = reshape(vecū,nx-1,ny-1)
    
            # X edges (arithmetic mean)
            ûx = ū[1:end-1,:] .* Δx[1:end-1] + ū[2:end,:] .* Δx[2:end]
            ûx ./= Δx[1:end-1] + Δx[2:end]
    
            # Y edges (arithmetic mean)
            ûy = ū[:,1:end-1] .* Δy[1:end-1]' + ū[:,2:end] .* Δy[2:end]'
            ûy ./= Δy[1:end-1]' + Δy[2:end]'
    
            # Transport flux in each direction
            f̂x = Mx .* ûx
            f̂y = My .* ûy
    
            # Viscosity in x direction
            ĝx = -β * diff(ū,dims=1) ./ Δx̄

            # Diffisive flux in each direction
            ĝy = -θ * diff(ū,dims=2) ./ Δȳ'
    
            # Total flux
            xflux = f̂x + ĝx
            yflux = f̂y + ĝy
        
            dū = 0.0ū
            dū[1:end-1,:] -= xflux ./ Δx[1:end-1]
            dū[2:end,:]   += xflux ./ Δx[2:end]
            dū[:,1:end-1] -= yflux ./ Δy[1:end-1]'
            dū[:,2:end]   += yflux ./ Δy[2:end]'
            
            return [dū[:]; sum(ū .* dA)]

        end

    ## Construct and solve linear system
        
        # Autodiff to get Jacobian
        J = sparse(ForwardDiff.jacobian(pde,vecū)[2:end,:])
        b = [zeros(size(J,1) - 1); 1.0]
        verbose && display("Jacobian constructed.")

        # Solve using GMRES
        vecu = gmres(J,b,vecū)[1]
        P = reshape(vecu,nx-1,ny-1)
        
        # Expectation function to return
        E(f,t=0.0) = sum([f(xᵢ,yᵢ) for xᵢ in x̄, yᵢ in ȳ] .* P .* dA)

    ## Return
    return P,x̄,ȳ,E

end


"""
    Initial condition guess
"""
function binary_ornstein_default_ic(p,d::Distribution,σ=0.1)
    x₀ = binary_equilibrium_proportion(p,mean(d))
    y₀ = 0.0
    MvNormal([x₀,y₀],[σ * x₀,1.0])
end

"""
    Default grid
"""
function binary_ornstein_default_xgrid(p,d,nx;threshold=-1)
    nx1 = Int(round(nx * 0.5))
    x1 = range(0.0,binary_equilibrium_proportion(p,d,threshold),nx1)
    x2 = x1[end] .+ exp.(range(log(diff(x1)[end]),log(1 - x1[end]),nx-nx1))
    [x1;x2]
end

########################################################
## MOMENT APPROACH (NORMAL ONLY)
########################################################

"""
    Create SDE model and moment equations
"""
function binary_normal_setup_moments(closure="zero")

    # Parameter handling...
    function pars(p,d,θ)
        # Parameters
        q,ω = p
        a,b = ω * (1 - q), ω * q
        [_θ => θ, _μ => mean(d), _σ => std(d), _a => a, _b => b]
    end

    # Setup SDE model
    @parameters _θ, _μ, _σ, _a, _b
    @variables t, _x(t), _y(t)

    _f = [
        Differential(t)(_x) ~ _b - _x * (_a + _b + (_μ + _σ * _y) * (1 - _x)),
        Differential(t)(_y) ~ -_θ * _y
    ]
    _g = [0; sqrt(2_θ)]

    sde_model = SDESystem(_f, _g, t, [_x,_y], [_θ,_μ,_σ,_a,_b], name = :model)

    # Setup moment equations
    moment_eqs = generate_raw_moment_eqs(sde_model,2)
    closed_eqs = moment_closure(moment_eqs, closure)

    ode_prob(p,d::Normal,θ,tend,ic=binary_ornstein_default_ic(p,d)) = ODEProblem(closed_eqs,moment_IC(ic,closed_eqs),(0.0,tend),pars(p,d,θ))

    return ode_prob
end
binary_normal_moments_ode = binary_normal_setup_moments()

"""
    Setup initial conditions for the moment equations based on a distribution (or not)
"""
function distribution_IC(X₀::Distribution,closed_eqs)
    return [
        closed_eqs.open_eqs.μ[(1,0)] => mean(X₀)[1],
        closed_eqs.open_eqs.μ[(0,1)] => mean(X₀)[2],
        closed_eqs.open_eqs.μ[(2,0)] => cov(X₀)[1,1] + mean(X₀)[1]^2,
        closed_eqs.open_eqs.μ[(1,1)] => cov(X₀)[1,2] + prod(mean(X₀)),
        closed_eqs.open_eqs.μ[(0,2)] => cov(X₀)[2,2] + mean(X₀)[2]^2
    ]
end
moment_IC(X₀::Distribution,eqs) = distribution_IC(X₀,eqs)
moment_IC(X₀::Vector,eqs) = deterministic_IC(X₀,eqs)

"""
    Steady-state from moment closure approximation
"""
function solve_binary_ornstein_stationary_moments(p,d::Normal,θ,ic=binary_ornstein_default_ic(p,d))
    prob = binary_normal_moments_ode(p,d,θ,0.0,ic)
    solve(SteadyStateProblem(prob)).u
    #Ex,Ey,Exx,Exy,Eyy
end

########################################################
## FITNESS
########################################################

"""
    Automatic dispatch to calculate fitness (with error, if applicable)
"""
function solve_binary_ornstein_fitness(p::Vector,d,θ;kwargs...)
    # QSS limit
    θ == 0.0 && return solve_binary_ornstein_fitness_qss(p,d)
    # Fast limit
    isinf(θ) && return solve_binary_ornstein_fitness_fast(p,d)
    # General
    P,x,y,E = solve_binary_ornstein(p,d,θ;kwargs...)
    solve_binary_ornstein_fitness(d,E,0.0)
end

""" 
    Time-varying fitness (general) for the Ornstein-Uhlenbeck driven model.
"""
function solve_binary_ornstein_fitness(d::Distribution,E)
    t -> E((x,y) -> (1 - x) * link_fcn(d,y),t)
end

"""
    Estimate stationary fitness using the end point. Additionally returns
    fitness gradient, and warns if non-convergence is apparent.
"""
function solve_binary_ornstein_fitness(d::Distribution,E,tend)
    λfcn = solve_binary_ornstein_fitness(d,E)
    λ = λfcn(tend)
    λ′ = ForwardDiff.derivative(λfcn,tend)
    if abs(λ′ / λ) > 1e-4
        @warn("May not have converged. Fitness gradient is $λ′ (fitness = $λ)")
    end
    return (λ,λ′)
end

"""
    QSS regime (θ → 0)
"""
function solve_binary_ornstein_fitness_qss(p,d;q=1e-6)
    # Proportion is at equilibrium always   
    x(λ) = binary_equilibrium_proportion(p,λ)
    # Fitness as a function of current growth rate
    func(λ) = (1 - x(λ)) * λ
    # Expectation
    lims = quantile.(d,[q,1 - q])
    quadgk(λ -> func(λ) * pdf(d,λ),lims...)[1]
end

""" 
    Fast regime (θ → ∞)
"""
function solve_binary_ornstein_fitness_fast(p,d)
    # Proportion is at equilibrium always   
    x = binary_equilibrium_proportion(p,mean(d))
    # Fitness
    (1 - x) * mean(d)
end

"""
    Fitness approximation using moment approach (general time)
"""
function solve_binary_ornstein_fitness_moments(p,d,θ,tend::Number,ic=binary_ornstein_default_ic(p,d))
    prob = binary_normal_moments_ode(p,d,θ,tend,ic)
    sol = solve(prob)
    Ex = t -> sol(t)[1]
    Ey = t -> sol(t)[2]
    Exy = t -> sol(t)[4]
    t -> mean(d) + std(d)*Ey(t) - mean(d)*Ex(t) - std(d)*Exy(t)
end

"""
    Fitness approximation using moment approach (stationary)
"""
function solve_binary_ornstein_fitness_moments(p,d,θ,ic=binary_ornstein_default_ic(p,d))
    Ex,Ey,Exx,Exy,Eyy = solve_binary_ornstein_stationary_moments(p,d,θ,ic)
    mean(d) + std(d)*Ey - mean(d)*Ex - std(d)*Exy
end