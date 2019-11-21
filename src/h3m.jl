struct H3M{T}
    M::Vector{T}
    ω::Vector{Float64}
end

# TODO: argcheck length
# TODO: isprobvec check

length(m::H3M) = length(m.M)

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

function vhem_step(base::H3M{Z}, reduced::H3M{Z}, τ::Integer, N::Integer) where Z
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
                    lgmm[i,j][β,ρ], logη[i,j][β,ρ] = loglikelihood_va(Miβ, Mjρ)
                end
            end

            lhmm[i,j], logϕ, logϕ1 = loglikelihood_va(Mi, Mj, lgmm[i,j], τ)
            
            # Initial probabilities (π in the paper)
            # Transition matrices (a or A in the paper)
            logai = log.(Mi.a)
            logAi = log.(Mi.A)

            ## Summary statistics
            logν[i,j] = zeros(τ, Kj, Ki)
            logξ[i,j] = zeros(τ, Kj, Kj, Ki)
            
            ## Aggregate summaries
            νagg1[i,j] = zeros(Kj)
            νagg[i,j]  = zeros(Kj,Ki)
            ξagg[i,j]  = zeros(Kj,Kj)

            for ρ in OneTo(Kj), β in OneTo(Ki)
                logν[i,j][1,ρ,β] = logai[β] + logϕ1[β,ρ]
                νagg1[i,j][ρ]  += exp(logν[i,j][1,ρ,β])
                νagg[i,j][ρ,β] += exp(logν[i,j][1,ρ,β])
            end

            for t in 2:τ
                for ρ in OneTo(Kj)
                    for β in OneTo(Ki)
                        for ρp in OneTo(Kj)
                            # TODO: Memory access patterns!
                            logtmp = logsumexp([logν[i,j][t-1,ρp,βp] + logAi[βp,β] for βp in OneTo(Ki)])
                            logξ[i,j][t,ρp,ρ,β] = logtmp + logϕ[t,β,ρp,ρ]
                            ξagg[i,j][ρp,ρ] += exp(logξ[i,j][t,ρp,ρ,β])
                        end
                        logν[i,j][t,ρ,β] = logsumexp(logξ[i,j][t,:,ρ,β])
                        νagg[i,j][ρ,β] += exp(logν[i,j][t,ρ,β])
                    end
                end
            end
        end
    end
    
    # Compute optimal assignment probabilities
    logz = zeros(length(base.M), length(reduced.M))

    for (i, ωi) in enumerate(base.ω)
        for (j, ωj) in enumerate(reduced.ω)
            logz[i,j] = log(ωj) + (N * ωi * lhmm[i,j])
        end
        logz[i,:] .-= logsumexp(logz[i,:])
    end

    # TODO: Use log below instead ?
    # This would require to compute aggregate statistics in log also...
    z = exp.(logz)

    ## M-step

    # 1. Compute H3M weights
    newω = zeros(length(reduced))
    norm = 0.0
    
    for j in OneTo(length(reduced)), i in OneTo(length(base))
        newω[j] += z[i,j]
        norm += z[i,j]
    end

    for j in OneTo(length(reduced))
        newω[j] /= norm
    end

    # 2. Compute H3M models
    newM = Vector{Z}(undef, length(reduced))

    for j in OneTo(length(reduced))
        Mj, ωj = reduced.M[j], reduced.ω[j]
        Kj = size(Mj, 1)

        # 2.a Compute initial probabilities
        newa = zeros(Kj)
        norm = 0.0

        for ρ in OneTo(Kj)
            for i in OneTo(length(base))
                newa[ρ] += z[i,j] * base.ω[i] * νagg1[i,j][ρ]
            end
            norm += newa[ρ]
        end

        newa /= norm

        # 2.b Compute transition matrix
        newA = zeros(Kj, Kj)

        for ρ in OneTo(Kj)
            norm = 0.0
            for ρp in OneTo(Kj)
                for i in OneTo(length(base))
                    newA[ρ,ρp] += z[i,j] * base.ω[i] * ξagg[i,j][ρ,ρp]
                end
                norm += newA[ρ,ρp]
            end

            for ρp in OneTo(Kj)
                newA[ρ,ρp] /= norm
            end
        end

        # 2.c Compute observation mixtures
        newB = Vector{UnivariateDistribution}(undef, Kj)

        for (ρ, Mjρ) in enumerate(Mj.B)
            norm = 0.0
            newc = zeros(ncomponents(Mjρ))
            newd = Vector{Normal}(undef, length(Mj.B[ρ].components))

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

                newμ  /= newc[l]
                newσ2 /= newc[l]
                norm  += newc[l]

                newd[l] = Normal(newμ, sqrt(newσ2))
            end
            
            newc /= norm
            newB[ρ] = MixtureModel(newd, newc)
        end

        # 2.d Build HMM
        newM[j] = HMM(newa, newA, newB)
    end

    H3M(newM, newω), lhmm, z
end