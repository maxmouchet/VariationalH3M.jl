
function lowerbound_mc(a::Distribution, b::Distribution, N = 10^6)
    mean(logpdf.(b, rand(a, N)))
end

function lowerbound_mc(a::HMM, b::HMM, N, T)
    mean([forward(b, rand(a, T), logl = true, robust = true)[2] for _ in 1:N])
end
