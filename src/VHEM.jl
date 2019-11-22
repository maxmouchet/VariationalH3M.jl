module VHEM

using ArgCheck
using Distributions
using StatsFuns
using HMMBase

import Base: OneTo, length

export loglikelihood_mc, loglikelihood_va, vhem_step, vhem_step_E, H3M

include("mc.jl")
include("va.jl")
include("lse.jl")
include("h3m.jl")

end