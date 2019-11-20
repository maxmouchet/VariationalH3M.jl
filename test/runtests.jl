using Test
using Distributions
using HMMBase
using VHEM

import Base: rand

rand(::Type{Normal}) = Normal(rand(Normal(0, 10)), rand(Gamma(1, 2)))

function rand(::Type{MixtureModel})
    K = rand(1:10)
    p = rand(Dirichlet(K, 1.0))
    c = [rand(Normal) for _ in 1:K]
    MixtureModel(c, p)
end

function rand(::Type{HMM})
    K = rand(1:10)
    a = rand(Dirichlet(K, 1.0))
    A = randtransmat(K, 1.0)
    B = [rand(MixtureModel) for _ in 1:K]
    HMM(a, A, B)
end

@testset "lowerbound ::Normal" begin
    @test lowerbound(Normal(0,1), Normal(0,1)) == -1.4189385332046727

    for _ in 1:100
        a = rand(Normal)
        b = rand(Normal)

        lb_mc = lowerbound_mc(a, b)
        lb_va = lowerbound(a, b)

        # 0.5%
        tol = (0.5 / 100) * abs(lb_mc)
        @test lb_va ≈ lb_mc atol=tol
    end
end

@testset "lowerbound ::MixtureModel" begin
    @test lowerbound(MixtureModel([Normal(0,1)]), MixtureModel([Normal(0,1)])) == -1.4189385332046727

    for _ in 1:10
        a = rand(MixtureModel)
        b = rand(MixtureModel)

        lb_mc = lowerbound_mc(a, b, 10^5)
        lb_va = lowerbound(a, b)
        
        @show lb_mc, lb_va

        # # 1%
        # tol = (1.0 / 100) * abs(lb_mc)
        # @test lb_va ≈ lb_mc atol=tol
    end
end

@testset "lowerbound ::HMM" begin
    for _ in 1:10
        a = rand(HMM)
        b = rand(HMM)

        lb_mc = lowerbound_mc(a, b, 10^3, 10^2)
        lb_va, _, _ = lowerbound(a, b, 10^2)

        @show lb_mc, lb_va

        # # 1%
        # tol = (1.0 / 100) * abs(lb_mc)
        # @test lb_va ≈ lb_mc atol=tol
    end
end