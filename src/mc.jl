# Monte Carlo expectations

function loglikelihood_mc(a::Distribution, b::Distribution, N::Integer)
    mean(logpdf.(b, rand(a, N)))
end

function loglikelihood_mc(a::HMM, b::HMM, τ::Integer, N::Integer)
    mean([
        HMMBase.loglikelihood(b, rand(a, τ), logl = true, robust = true)
        for _ in 1:N
    ])
end
