mutable struct LogSumExpAcc{T<:Number}
    m::T # Current maximum value
    s::T # Sum
end

LogSumExpAcc() = LogSumExpAcc(-Inf, 0.0)

function add!(acc::LogSumExpAcc{T}, val::T) where T
    # TODO: Re-evaluate only if causing overflow
    if val <= acc.m
        acc.s += exp(val - acc.m)
    else
        acc.s *= exp(acc.m - val)
        acc.s += 1.0
        acc.m = val
    end
end

function add!(acc::LogSumExpAcc{T}, vals::Vector{T}) where T
    for val in vals
        add!(acc, val)
    end
end

sum(acc::LogSumExpAcc{T}) where T = log(acc.s) + acc.m
