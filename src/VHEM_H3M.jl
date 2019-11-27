module VHEM_H3M

using ArgCheck
using Distributions
using StatsFuns
using HMMBase

import Base: length, iterate, sum, OneTo

export
    # mc.jl
    loglikelihood_mc,
    # va.jl
    loglikelihood,
    loglikelihood_va,
    # h3m.jl
    H3M,
    # api.jl
    cluster

include("mc.jl")
include("va.jl")
include("lse.jl")
include("h3m.jl")
include("em.jl")
include("api.jl")

end