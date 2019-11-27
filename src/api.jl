
function loglikelihood(base::H3M, reduced::H3M, z::AbstractMatrix, lhmm::AbstractMatrix, N)
    sum(OneTo(length(base))) do i
        sum(OneTo(length(reduced))) do j
            z[i,j] * (log(reduced.ω[j] / (z[i,j] + eps())) + N*lhmm[i,j])
        end
    end
end

function cluster(base::H3M, reduced::H3M, τ::Integer, N::Integer; maxiter = 100, tol = 1e0)
    @argcheck maxiter >= 0

    z = zeros(length(base), length(reduced))
    history = EMHistory(false, 0, [])
    logtot = -Inf

    for it in 1:maxiter
        reduced, lhmm, z = vhem_step(base, reduced, τ, N)
        logtotp = loglikelihood(base, reduced, z, lhmm, N)

        push!(history.logtots, logtotp)
        history.iterations += 1

        if abs(logtotp - logtot) < tol
            history.converged = true
            break
        end

        logtot = logtotp
    end

    history, z, reduced
end

mutable struct EMHistory
    converged::Bool
    iterations::Int
    logtots::Vector{Float64}
end