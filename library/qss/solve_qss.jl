using Roots
using QuadGK

"""
    Solve QSS for a fixed environment
"""
function solve_qss(λ::Function,p̂::Distribution,ω;ret_pdf=false,ret_resid=false,eps=1e-10,λrange=:auto)

    # Extract components from mixture (or not...)
    v = isa(p̂,MixtureModel) ? p̂.components : [p̂]
    w = isa(p̂,MixtureModel) ? p̂.prior.p : [1.0]

    # λ must be monotonic. Translate such that minimum is zero.
    λ̂ = ε -> λ(ε) - min(λ(0),λ(1))

    # Scaling function (as a function of Ê...)
    r = (ε,Ê) -> -ω / (λ̂(ε) - Ê - ω)

    # Setup residual function to solve using root finding
    function residual(Ê)

        # Expectation w.r.t. each element of mixture
        function Êᵢ(vᵢ)
            if var(vᵢ) < 1e-10
                return λ̂(mean(vᵢ)) * r(mean(vᵢ),Ê)
            else
                return quadgk(ε -> λ̂(ε) * r(ε,Ê) * pdf(vᵢ,ε),0.0,1.0)[1]
            end
        end
    
        # Get residual expectation
        return Ê - sum(Êᵢ(v[i]) * w[i] for i in eachindex(v))
        
    end
    ret_resid && return residual

    # Bounds on the expectation
    if λrange == :auto
        maxλ̂ = max(λ̂(0),λ̂(1))
        a,b = (maxλ̂ - ω + eps,maxλ̂)
    else
        a,b = λrange
    end
    
    # Find residual
    #! check this logic later (using stability of mass solution)
    if sign(residual(a)) == sign(residual(b))
        Ê = -ω - min(λ(0),λ(1))
    else
        Ê = find_zero(residual,(a,b))
    end

    # Return, or calculate distribution pdf if necessary...
    if !ret_pdf
        return Ê + min(λ(0),λ(1))
    else
        if !isa(p̂,MixtureModel)
            return ε -> r(ε,Ê) * pdf(v[1],ε)
        else
            ṽ = [ε -> r(ε,Ê) * pdf(vᵢ,ε) for vᵢ in v]
            p̃ = ε -> sum(w[i] * ṽ[i](ε) for i = eachindex(w))
            return p̃
        end
    end

end

"""
    Solve (expected) QSS fitness in a random environment
"""
function solve_qss_fitness(λ::Function,p̂::Distribution,ω::Number,d=Normal();q=1e-6,λrange=:auto)
    
    # Function to integrate
    func = z -> solve_qss(ε -> λ(ε,z),p̂,ω;λrange) * pdf(d,z)

    # Integrate (or sum)
    if isa(d,ContinuousDistribution)
        lims = quantile.(d,[q,1-q])
        return quadgk(func,lims...)[1]
    elseif isa(d,DiscreteDistribution)
        return sum(func(z) for z in support(d))
    end

end

