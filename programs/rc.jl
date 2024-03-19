using LinearAlgebra
using SparseArrays
using Distributions
using ArnoldiMethod

struct rc
    N::Int64
    Nu::Int64
    Ny::Int64

    α::Float64
    β::Float64

    r::Vector{Float64}
    b::Vector{Float64}
    W::SparseMatrixCSC{Float64, Int64}
    Wi::Matrix{Float64}
    Wo::Matrix{Float64}

    Φ::Function

    function rc(N, Nu, Ny; ρ, R, σi, σb, α, β, Φ=tanh)
        @assert R > 0

        rfn(N) = rand(Uniform(-1, 1), N)
        W = sprand(N, N, ρ, rfn)
        λ, φ = partialeigen(partialschur(W, nev=1, tol=1e-3, which=LM())[1])
        W = W .* (R / abs(λ[1]))

        r = zeros(Float64, N)
        b = rand(Uniform(-σb, σb), N)
        Wi = rand(Uniform(-σi, σi), (N, Nu))
        Wo = Matrix{Float64}(undef, (Ny, N))

        new(N, Nu, Ny, α, β, r, b, W, Wi, Wo, Φ)
    end
end

function r_update!(rc::rc, rtp1, rt, ut, tmp)
    mul!(tmp, rc.W, rt)
    mul!(tmp, rc.Wi, ut, 1, 1)
    rtp1 .= rc.α.*rc.Φ.(tmp .+ rc.b) .+ (1-rc.α).*rt
end

function rc_drive(rc::rc, u::AbstractMatrix)
    @assert size(u)[1] == rc.Nu

    n_steps = size(u)[2]
    r = Matrix{Float64}(undef, (rc.N, n_steps))
    tmp = Vector{Float64}(undef, rc.N)

    @views r_update!(rc, r[:, 1], rc.r, u[:, 1], tmp)
    @inbounds for t in 2:n_steps
        @views r_update!(rc, r[:, t], r[:, t-1], u[:, t], tmp)
    end

    rc.r .= r[:, n_steps]
    return r
end

function rc_train(rc::rc, u::AbstractMatrix, y::AbstractMatrix)
    @assert size(u)[2] == size(y)[2]
    r = rc_drive(rc, u)
    rc.Wo[:, :] .= y*r'/(r*r' + rc.β*I)#(Symmetric(r*r'+rc.β*I)\(r*y'))'
end

function rc_drive_output(rc::rc, u::AbstractMatrix)
    n_steps = size(u)[2]
    r = Matrix{Float64}(undef, (rc.N, n_steps))
    y = Matrix{Float64}(undef, (rc.Ny, n_steps))
    tmp = Vector{Float64}(undef, rc.N)

    @views r_update!(rc, r[:, 1], rc.r, u[:, 1], tmp)
    @views mul!(y[:, 1], rc.Wo, rc.r)
    @inbounds for t in 2:n_steps
        @views r_update!(rc, r[:, t], r[:, t-1], u[:, t], tmp)
        @views mul!(y[:, t], rc.Wo, r[:, t])
    end

    rc.r .= r[:, n_steps]
    return r, y
end

function rc_drive_forecast(rc::rc, n_steps::Int)
    r = Matrix{Float64}(undef, (rc.N, n_steps))
    y = Matrix{Float64}(undef, (rc.Ny, n_steps))
    tmp = Vector{Float64}(undef, rc.N)

    r[:, 1] .= rc.r
    @views mul!(y[:, 1], rc.Wo, rc.r)
    @inbounds for t in 2:n_steps
        @views r_update!(rc, r[:, t], r[:, t-1], y[:, t-1], tmp)
        @views mul!(y[:, t], rc.Wo, r[:, t])
    end

    rc.r .= r[:, n_steps]
    return r, y
end
