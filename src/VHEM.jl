module VHEM

using Distributions
using StatsFuns
using HMMBase

export lowerbound, lowerbound_mc, vardists, vhem_step, H3M

"""
Expectation of a gaussian wrt. another gaussian
(Penny and Roberts, 2000)
E_{a} L(b)
"""
function lowerbound(a::Normal, b::Normal)
    -(1/2) * (log2π + log(b.σ^2) + (a.σ^2 / b.σ^2) + ((b.μ - a.μ)^2 / b.σ^2))
end

function vardists(a::MixtureModel, b::MixtureModel)
    I, J = ncomponents(a), ncomponents(b)
    
    # Expected log-likelihoods
    E = zeros(I, J)
    for (i, ai) in enumerate(a.components)
        for (j, bj) in enumerate(b.components)
            E[i,j] = lowerbound(ai, bj)
        end
    end

    # η[i,j]: probability that an observation from
    # component i of a corresponds to component j of b
    # ηl = log η
    ηl = zeros(I, J)

    for i in 1:I
        for j in 1:J
            ηl[i,j] = log(b.prior.p[j]) + E[i,j]
        end
        ηl[i,:] .-= logsumexp(ηl[i,:])
    end

    E, ηl
end

"""
Variational lower bound to the expected log-likelihood.

Coviello 14, Hershey 07.
"""
function lowerbound(a::MixtureModel, b::MixtureModel, E::AbstractMatrix, ηl::AbstractMatrix)
    I, J = size(ηl)

    # Optimal lower bound to expected log-likelihood
    LL = 0.0
    for i in 1:I
        ll = 0.0
        # TODO: logsumexp instead ?
        for j in 1:J
            ll += exp(ηl[i,j]) * (log(b.prior.p[j]) - ηl[i,j] + E[i,j])
        end
        LL += a.prior.p[i] * ll
    end
    
    LL
end

function lowerbound(a::MixtureModel, b::MixtureModel)
    E, ηl = vardists(a, b)
    lowerbound(a, b, E, ηl)
end

"""
Variational lower bound to the expected log-likelihood.

Coviello 14, Hershey 07.
"""
function lowerbound(hmm1::HMM, hmm2::HMM, τ::Integer)
    # TODO: Scaling / Numerical stability of this
    K, L = size(hmm1, 1), size(hmm2, 1)
    
#     # Lower-bound on the states observations distributions log-likelihood
#     LLO = zeros(K, L)
#     for i in 1:K, j in 1:L
#         LLO[i,j] = lowerbound(hmm1.B[i], hmm2.B[j])
#     end
    
#     # zt-1 -> zt, ii -> i, jj -> j
#     LL = zeros(T+1, K, L)
#     ϕl = zeros(T, L, K, L)

#     for t in T:-1:2
#         for ii in 1:K
#             for jj in 1:L
#                 for i in 1:K
#                     ϕl[t,ii,]
#                 end
#             end
#         end
#     end

    LGMM = zeros(K, L)
    for β in 1:K, ρ in 1:L
        LGMM[β,ρ] = lowerbound(hmm1.B[β], hmm2.B[ρ])
    end

    LL = zeros(τ+1, K, L)
    for t in τ:-1:2
        for βp in 1:K
            for ρp in 1:L
                for β in 1:K
                    # TODO: Reuse ϕl instead, and compute in a single pass
                    ll = logsumexp([log(hmm2.A[ρp,ρ]) + LGMM[β,ρ] + LL[t+1, β, ρ] for ρ in 1:L])
                    LL[t, βp, ρp] += hmm1.A[βp,β] * ll
                end
            end
        end
    end

    ϕl = zeros(τ, L, K, L)
    for t in τ:-1:2
        for ρp in 1:L
            for β in 1:K
                for ρ in 1:L
                    ϕl[t,ρp,β,ρ] = log(hmm2.A[ρp,ρ]) + LGMM[β,ρ] + LL[t+1, β, ρ]
                end
                ϕl[t,ρp,β,:] .-= logsumexp(ϕl[t,ρp,β,:])
            end
        end
    end

    ϕl1 = zeros(K, L)
    for β in 1:K
        for ρ in 1:L
            ϕl1[β,ρ] = log(hmm2.a[ρ]) + LGMM[β,ρ] + LL[2, β, ρ]
        end
        ϕl1[β,:] .-= logsumexp(ϕl1[β,:])
    end
    
    LLfinal = 0.0
    for β in 1:K
        ll = logsumexp([log(hmm2.a[ρ]) + LGMM[β,ρ] + LL[2, β, ρ] for ρ in 1:L])
        LLfinal += hmm1.a[β] * ll
    end
    
    LLfinal, ϕl, ϕl1
end

function lowerbound_mc(a::Distribution, b::Distribution, N = 10^6)
    mean(logpdf.(b, rand(a, N)))
end

function lowerbound_mc(a::HMM, b::HMM, N, T)
    mean([forward(b, rand(a, T), logl = true, robust = true)[2] for _ in 1:N])
end

struct H3M
    M::Vector{HMM}
    ω::Vector{Float64}
end

function Ω(f, b::H3M, j::Integer, ρ::Integer, z::AbstractMatrix, νagg)
    tot = 0.0
    for (i, ωi) in enumerate(b.ω)
        s = 0.0
        for β in 1:size(b.M[i],1)
            ss = 0.0
            for (m, c) in enumerate(b.M[i].B[β].prior.p)
                ss += c * f(i, β, m)
            end
            s += νagg[i,j][ρ,β] * ss
        end
        tot += z[i,j] * s
    end
    tot
end

function vhem_step(base::H3M, reduced::H3M, τ::Integer, N::Integer)
    ## Expectations
    # logη[i,j][β,ρ][m,l]
    logη = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int}, Matrix{Float64}}}()
    
    # lgmm[i,j][β,ρ]: expected log-likelihood
    lgmm = Dict{Tuple{Int,Int}, Matrix{Float64}}()
    
    # lhmm[i,j]: expected log-likelihood
    lhmm = zeros(length(base.M), length(reduced.M))

    ## Summary statistics
    # logν[i,j][t,ρ,β]
    logν = Dict{Tuple{Int,Int}, Array{Float64,3}}()

    # ξ[i,j][t,ρp,ρ,β]
    logξ = Dict{Tuple{Int,Int}, Array{Float64,4}}()

    # νagg1[i,j][ρ]
    νagg1 = Dict{Tuple{Int,Int}, Vector{Float64}}()
    
    # νagg[i,j][ρ,β]
    νagg = Dict{Tuple{Int,Int}, Matrix{Float64}}()

    # ξagg[i,j][ρ,ρ']
    ξagg = Dict{Tuple{Int,Int}, Matrix{Float64}}()

    ## E-step
    
    for (i, Mi) in enumerate(base.M)
        for (j, Mj) in enumerate(reduced.M)
            Ki, Kj = size(Mi, 1), size(Mj, 1)

            ## Expectations
            logη[i,j] = Dict{Tuple{Int,Int}, Matrix{Float64}}()
            lgmm[i,j] = zeros(Ki, Kj)

            for (β, Miβ) in enumerate(Mi.B)
                for (ρ, Mjρ) in enumerate(Mj.B)
                    E, logη[i,j][β,ρ] = vardists(Miβ, Mjρ)
                    lgmm[i,j][β,ρ] = lowerbound(Miβ, Mjρ, E, logη[i,j][β,ρ])
                end
            end
            
            # TODO: Optimize by not recomputing lgmm, ...
            lhmm[i,j], logϕ, logϕ1 = lowerbound(Mi, Mj, τ)
            
            ## Summary statistics
            logν[i,j]  = zeros(τ, Kj, Ki)
            logξ[i,j]  = zeros(τ, Kj, Kj, Ki)
            
            ## Aggregate summaries
            νagg1[i,j] = zeros(Kj)
            νagg[i,j]  = zeros(Kj,Ki)
            ξagg[i,j]  = zeros(Kj,Kj)

            for ρ in 1:Kj, β in 1:Ki
                logν[i,j][1,ρ,β] = log(Mi.a[β]) + logϕ1[β,ρ]
            end
            
            for t in 2:τ
                for ρp in 1:Kj, ρ in 1:Kj, β in 1:Ki
                    logtmp = logsumexp([logν[i,j][t-1,ρp,βp] + log(Mi.A[βp,β]) for βp in 1:Ki])
                    logξ[i,j][t,ρp,ρ,β] = logtmp + logϕ[t,ρp,β,ρ]
                end
                
                for ρ in 1:Kj, β in 1:Ki
                    logν[i,j][t,ρ,β] = logsumexp(logξ[i,j][t,:,ρ,β])
                end
            end
            
            for ρ in 1:Kj
                νagg1[i,j][ρ] = sum(exp.(logν[i,j][1,ρ,:]))
            end
            
            for ρ in 1:Kj, β in 1:Ki
                νagg[i,j][ρ,β] = sum(exp.(logν[i,j][:,ρ,β]))
            end

            for ρp in 1:Kj, ρ in 1:Kj
                ξagg[i,j][ρp,ρ] = sum(exp.(logξ[i,j][:,ρp,ρ,:]))
            end
        end
    end
    
    # Compute optimal assignment probabilities
    z = zeros(length(base.M), length(reduced.M))

    for (i, ωi) in enumerate(base.ω)
        for (j, ωj) in enumerate(reduced.ω)
            z[i,j] = ωj * exp(N * ωi * lhmm[i,j])
        end
        z[i,:] ./= sum(z[i,:])
    end
    
    ## M-step
    
    # 1. Compute H3M weights
    newω = sum(z, dims = 1)[:]
    newω ./= sum(newω)
    
    # 2. Compute H3M models
    newM = HMM[]

    for (j, (Mj, ωj)) in enumerate(zip(reduced.M, reduced.ω))
        Kj = size(Mj,1)
        
        # 2.a Compute initial probabilities
        newa = zeros(Kj)
        for ρ in 1:Kj, (i, ωi) in enumerate(base.ω)
            newa[ρ] += z[i,j] * ωi * νagg1[i,j][ρ]
        end
        newa ./= sum(newa)
        
        # 2.b Compute transition matrix
        newA = zeros(Kj, Kj)

        for ρ in 1:Kj
            for ρp in 1:Kj, (i, ωi) in enumerate(base.ω)
                newA[ρ,ρp] += z[i,j] * ωi * ξagg[i,j][ρ,ρp]
            end
            newA[ρ,:] ./= sum(newA[ρ,:])
        end
        
        # 2.c Compute observation mixtures
        newB = UnivariateDistribution[]

        for (ρ, Mjρ) in enumerate(Mj.B)
            newc = zeros(ncomponents(Mjρ))
            newd = Normal[]
    
            for (l, Mjρl) in enumerate(Mj.B[ρ].components)
                newc[l] = Ω(base, j, ρ, z, νagg) do i, β, m
                    exp.(logη[i,j][β,ρ][m,l])
                end
                
                newμ = Ω(base, j, ρ, z, νagg) do i, β, m
                    exp.(logη[i,j][β,ρ][m,l]) * base.M[i].B[β].components[m].μ
                end
                
                newσ2 = Ω(base, j, ρ, z, νagg) do i, β, m
                    exp.(logη[i,j][β,ρ][m,l]) * (base.M[i].B[β].components[m].σ^2 + (base.M[i].B[β].components[m].μ - Mjρl.μ)^2)
                end

                newμ /= newc[l]
                newσ2 /= newc[l]

                push!(newd, Normal(newμ, sqrt(newσ2)))
            end
            
            newc /= sum(newc)
            push!(newB, MixtureModel(newd, newc))
        end
        
        # 2.d Build HMM
        push!(newM, HMM(newa, newA, newB))
    end

    H3M(newM, newω), lhmm, z
end

end