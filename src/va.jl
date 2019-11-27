# Variational approximations

# Not an approximation

"""
Expectation of a gaussian wrt. another gaussian
(Penny and Roberts, 2000)
E_{a} L(b)
"""
function loglikelihood(a::Normal, b::Normal)
    if iszero(b.σ)
        (a.μ == b.μ) ? Inf : 0
    else
        -(1/2) * (log2π + log(b.σ^2) + (a.σ^2 / b.σ^2) + ((b.μ - a.μ)^2 / b.σ^2))
    end
end

function loglikelihood_va(a::MixtureModel, b::MixtureModel)
    I, J = ncomponents(a), ncomponents(b)

    # Optimal lower-bound to the expected log-likelihood
    lb = 0.0

    # Expected log-likelihoods
    logl = Matrix{Float64}(undef, (I, J))

    # η[i,j]: probability that an observation from
    # component i of a corresponds to component j of b
    logη = Matrix{Float64}(undef, (I, J))

    # Type annotation is important for performane
    # otherwiser the compiler infers `Any`.
    logp::Vector{Float64} = log.(b.prior.p::Vector{Float64})

    for (i, ai) in enumerate(a.components)
        for (j, bj) in enumerate(b.components)
            logl[i,j] = loglikelihood(ai, bj)
            logη[i,j] = logp[j] + logl[i,j]
        end
        logη[i,:] .-= logsumexp(logη[i,:])
    end

    for i in OneTo(I)
        p::Float64 = (a.prior.p::Vector{Float64})[i]
        for j in OneTo(J)
            lb += p * exp(logη[i,j]) * (logp[j] - logη[i,j] + logl[i,j])
        end
    end

    lb, logη
end

"""
Variational lower bound to the expected log-likelihood.

Coviello 14, Hershey 07.
"""
function loglikelihood_va(hmm1::HMM, hmm2::HMM, lgmm::Matrix{Float64}, τ::Integer)
    @argcheck size(hmm1, 1) == size(lgmm, 1)
    @argcheck size(hmm2, 1) == size(lgmm, 2)
    @argcheck τ >= 0

    K, L = size(hmm1, 1), size(hmm2, 1)

    # Optimal lower-bound to the expected log-likelihood
    lhmm = 0.0

    # logl[t,βp,ρp]
    logl = zeros(τ+1, K, L)

    # logϕ[t,β,ρp,ρ]
    logϕ = zeros(τ, K, L, L)

    # logϕ1[β,ρ]
    # TODO: Merge with logϕ (ignore previous state for t = 1) ?
    logϕ1 = zeros(K, L)

    # Initial probabilities (π in the paper)
    # Transition matrices (a or A in the paper)
    loga2 = log.(hmm2.a)
    logA2 = log.(hmm2.A)

    # Appendix B. (p. 737)
    # β  = β_t
    # βp = β_{t-1}

    for t in τ:-1:2
        for β in 1:K
            for ρp in 1:L
                for ρ in 1:L
                    logϕ[t,β,ρp,ρ] = logA2[ρp,ρ] + lgmm[β,ρ] + logl[t+1,β,ρ]
                end

                norm = logsumexp(logϕ[t,β,ρp,:])
                logϕ[t,β,ρp,:] .-= norm

                for βp in 1:K
                    logl[t,βp,ρp] += hmm1.A[βp,β] * norm
                end
            end
        end
    end

    for β in 1:K
        for ρ in 1:L
            logϕ1[β,ρ] = loga2[ρ] + lgmm[β,ρ] + logl[2,β,ρ]
        end

        norm = logsumexp(logϕ1[β,:])
        logϕ1[β,:] .-= norm

        lhmm += hmm1.a[β] * norm
    end

    lhmm, logϕ, logϕ1
end

function loglikelihood_va(hmm1::HMM, hmm2::HMM, τ::Integer)
    Ki, Kj = size(hmm1, 1), size(hmm2, 1)

    logη = Dict{Tuple{Int,Int}, Matrix{Float64}}()
    lgmm = zeros(Ki, Kj)

    for (β, Miβ) in enumerate(hmm1.B)
        for (ρ, Mjρ) in enumerate(hmm2.B)
            lgmm[β,ρ], logη[β,ρ] = loglikelihood_va(Miβ, Mjρ)
        end
    end

    logη, lgmm, loglikelihood_va(hmm1, hmm2, lgmm, τ)
end

