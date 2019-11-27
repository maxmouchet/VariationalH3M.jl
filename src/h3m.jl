"""
    Hidden Markov Mixture Model
"""
struct H3M{T <: AbstractHMM}
    M::Vector{T}
    ω::Vector{Float64}
    H3M{T}(M, ω) where T = assert_h3m(M, ω) && new(M, ω)
end

H3M(M::AbstractVector{T}, ω) where T  = H3M{T}(M, ω)
H3M(M::AbstractVector{T}) where T = H3M{T}(M, ones(length(M))/length(M))

length(m::H3M) = length(m.M)
iterate(m::H3M, args...) = iterate(zip(m.M, m.ω), args...)

function assert_h3m(M, ω)
    @argcheck length(M) == length(ω)
    @argcheck isprobvec(ω)
    return true
end