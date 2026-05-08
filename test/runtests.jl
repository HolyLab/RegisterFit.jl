import RegisterFit
using Test, Aqua, ExplicitImports, Documenter, CoordinateTransformations, Interpolations, ImageBase, ImageTransformations, LinearAlgebra
using RegisterCore

using RegisterUtilities

@testset "Doctests" begin
    DocMeta.setdocmeta!(RegisterFit, :DocTestSetup, :(using RegisterFit); recursive=true)
    doctest(RegisterFit; manual=false)
end

@testset "Aqua" begin
    Aqua.test_all(RegisterFit)
end

@testset "ExplicitImports" begin
    test_explicit_imports(RegisterFit)
end

@testset "qfit" begin
    denom = ones(11,11)
    Q = rand(Float64,2,2); Q = Q'*Q
    num = quadratic(11, 11, [1,-2], Q)
    E0, cntr, Qf = @inferred(RegisterFit.qfit(MismatchArray(num, denom), 1e-3))
    @test abs(E0) < eps()
    @test cntr ≈ [1,-2]
    @test Qf ≈ Q

    num = num.+5
    E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), 1e-3)
    @test E0 ≈ 5
    @test cntr ≈ [1,-2]
    @test Qf ≈ Q

    num = quadratic(11, 13, [2,-4], Q)
    thresh = 1e-3
    scale = rand(Float64,size(num)) .+ thresh
    denom = ones(size(num)).*scale
    @test all(denom .> thresh)
    num = num.*scale
    E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), thresh)
    @test abs(E0) < eps()
    @test cntr ≈ [2,-4]
    @test Qf ≈ Q

    # Degenerate solutions
    Q = [1 0; 0 0]
    denom = ones(13, 11)
    num = quadratic(13, 11, [2,-4], Q)
    E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), thresh)
    @test abs(E0) < eps()
    @test cntr[1] ≈ 2
    @test Qf ≈ Q
    a = rand(2).+0.1
    Q = a*a'
    num = quadratic(13, 11, [2,-4], Q)
    E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), thresh)
    @test abs(E0) < eps()
    @test abs(dot(cntr-[2,-4], a)) < eps()
    @test ≈(Qf, Q, atol=1e-12)

    # Settings with very few above-threshold data points
    # Just make sure there are no errors
    denom0 = ones(5,5)
    Q = rand(Float64,2,2); Q = Q'*Q
    num0 = quadratic(5, 5, [0,0], Q)
    denom = copy(denom0); denom[1:2,1] *= 100; denom[5,5] *= 100
    num = copy(num0); num[1:2,1] *= 100; num[5,5] *= 100
    thresh = 2
    E0, cntr, Qf = RegisterFit.qfit(MismatchArray(num, denom), thresh)

    ### qbuild
    A = RegisterFit.qbuild(2, [-1,1], [0.3 0; 0 0.5], (5,5))
    v1 = 0.3*((-5:5).+1).^2
    v2 = 0.5*((-5:5).-1).^2
    @test A.data ≈ v1.+v2'.+2
end

@testset "uisvalid and uclamp!" begin
    # maxshift must exceed register_half (0.5001) for the bounds to be non-trivial
    maxshift = (3, 3, 4)
    u_valid = reshape([1.0, -2.0, 0.5], 3, 1)
    @test RegisterFit.uisvalid(u_valid, maxshift)
    u_invalid = reshape([2.6, -2.0, 0.5], 3, 1)
    @test !RegisterFit.uisvalid(u_invalid, maxshift)

    u_clamp = reshape([5.0, -6.0, 0.1], 3, 1)
    RegisterFit.uclamp!(u_clamp, maxshift)
    @test abs(u_clamp[1]) < maxshift[1]
    @test abs(u_clamp[2]) < maxshift[2]
    @test abs(u_clamp[3]) < maxshift[3]
end

@testset "qfit edge cases" begin
    # All below threshold → zero return
    num = rand(5, 5)
    denom = zeros(5, 5)
    E0, c, Q = RegisterFit.qfit(MismatchArray(num, denom), 1.0)
    @test E0 == 0
    @test all(c .== 0)
    @test all(Q .== 0)
end

@testset "optimize_per_aperture" begin
    # 1D grid of 2D mismatch arrays; only first shift component is stored per aperture
    # argmin_mismatch trims edges, so use shifts within -1:1 for a 5×5 array
    Q = [1.0 0; 0 1.0]
    mm1 = MismatchArray(quadratic(5, 5, [1, 0], Q), ones(5, 5))
    mm2 = MismatchArray(quadratic(5, 5, [-1, 0], Q), ones(5, 5))
    mms = [mm1, mm2]
    u = RegisterFit.optimize_per_aperture(mms, 0.5)
    @test size(u) == (1, 2)
    @test u[1, 1] ≈ 1
    @test u[1, 2] ≈ -1
end

@testset "mms2fit!" begin
    # Grid dimensionality must match the shift dimensionality of each mismatch array
    Q = [1.0 0; 0 1.0]
    mm1 = MismatchArray(quadratic(5, 5, [1, -1], Q), ones(5, 5))
    mm2 = MismatchArray(quadratic(5, 5, [0,  1], Q), ones(5, 5))
    mm3 = MismatchArray(quadratic(5, 5, [-1, 0], Q), ones(5, 5))
    mm4 = MismatchArray(quadratic(5, 5, [1,  0], Q), ones(5, 5))
    mms = reshape([mm1, mm2, mm3, mm4], 2, 2)
    cs, Qs, mmis = RegisterFit.mms2fit!(mms, 0.5)
    @test size(cs) == (2, 2)
    @test cs[1, 1] ≈ [1.0, -1.0] atol=1e-10
    @test cs[2, 1] ≈ [0.0,  1.0] atol=1e-10
end

@testset "PAT" begin
    # Principal Axes Transformation
    fixed = zeros(10,11)
    fixed[2,3:7] .= 1
    fixed[3,2:8] .= 1
    moving = zeros(10,11)
    moving[3:7,8] .= 1
    moving[2:8,7] .= 1
    fmean, fvar = RegisterFit.principalaxes(fixed)
    tfm = RegisterFit.pat_rotation((fmean, fvar), moving)
    for i = 1:2
        S = tfm[i].linear
        @test abs(S[1,2]) ≈ 1
        @test abs(S[2,1]) ≈ 1
        @test abs(S[1,1]) < 1e-8
        @test abs(S[2,2]) < 1e-8
    end

    # Test the array-input convenience wrapper
    tfms2 = RegisterFit.pat_rotation(fixed, moving)
    @test length(tfms2) == length(tfm)
    for i in eachindex(tfm)
        @test tfms2[i].linear ≈ tfm[i].linear
    end

    F = meanfinite(abs.(fixed); dims = (1,2))[1]

    df = zeros(2)
    movinge = extrapolate(interpolate(moving, BSpline(Linear())), NaN)
    origin_dest = center(movinge)
    origin_src = center(fixed)
    for i = 1:2
        # mov = TransformedArray(movinge, tfm[i])
        translation = tfm[i].translation - tfm[i].linear*origin_dest + origin_src
        tform = AffineMap(tfm[i].linear,translation)
        df[i] = meanfinite(abs.(fixed-[movinge(tform([idx[1], idx[2]])...) for idx in CartesianIndices(fixed)]); dims = (1,2))[1]
    end
    @test minimum(df) < 1e-4*F
end
