module RegisterFit

using CenterIndexedArrays: CenterIndexedArrays, CenterIndexedArray
using CoordinateTransformations: CoordinateTransformations, AffineMap
using Interpolations: Interpolations
using LinearAlgebra: LinearAlgebra, Diagonal, Hermitian, I, cholesky, det, diag, dot, eigen, mul!, svd
using NLsolve: NLsolve, nlsolve
using Optim: Optim
using RegisterCore: RegisterCore, MismatchArray, NumDenom, argmin_mismatch, maxshift
using RegisterPenalty: RegisterPenalty, interpolate_mm!
using StaticArrays: StaticArrays, SArray, SVector, Size, StaticVector, similar_type
using Statistics: Statistics, mean
import Base.Cartesian: @nloops, @nexprs, @nref, @nif

export
    mismatch2affine,
    mms2fit!,
    optimize_per_aperture,
    pat_rotation,
    principalaxes,
    qbuild,
    qfit,
    uisvalid,
    uclamp!

"""
RegisterFit provides functions that compute affine transformations minimizing
image registration mismatch, given per-aperture mismatch data from `RegisterMismatch`.

## Global optimization

- [`mismatch2affine`](@ref): affine transform from mismatch data by least squares
- [`pat_rotation`](@ref): rigid alignment via a Principal Axes Transformation
- [`optimize_per_aperture`](@ref): naive per-aperture displacement search

## Utilities

- [`qfit`](@ref): fit a single aperture's mismatch to a quadratic form
- [`mms2fit!`](@ref): prepare an array of mismatch arrays for optimization
- [`qbuild`](@ref): reconstruct mismatch data from a quadratic form
- [`uclamp!`](@ref) and [`uisvalid`](@ref): enforce/check bounds on displacements
"""
RegisterFit

# For bounds constraints
const register_half = 0.5001
const register_half_safe = 0.51

"""
    tform = mismatch2affine(mms, thresh, knots)

Return an `AffineMap` that is a least-squares "best initial guess" for the
transformation minimizing the mismatch.

`mms` is an array of `MismatchArray`s (one per aperture, in the format returned
by `RegisterMismatch`). `thresh` is the denominator threshold that determines
which aperture regions have sufficient pixel/intensity overlap to be valid.
`knots` specifies the aperture centers (see `RegisterDeformation`).

The algorithm fits each aperture to a quadratic, then solves a global least-squares
problem over all apertures — guaranteeing a global solution at the cost of the
quadratic approximation. If `thresh` is too restrictive, it is halved up to three
times before raising an error.

# Returns
- `tform::AffineMap` — affine transformation (linear map + translation)

To refine `tform` beyond the quadratic approximation, see `optimize` in the
registration pipeline.
"""
function mismatch2affine(mms, thresh, knots)
    gridsize = size(mms)
    N = length(gridsize)
    mm = first(mms)
    T = eltype(eltype(mm))
    TFT = typeof(one(T) / 2)  # transformtype; it needs to be a floating-point type
    n = prod(gridsize)
    # Fit the parameters of each quadratic
    u0 = Vector{Any}(undef, n)
    Q = Vector{Any}(undef, n)
    i = 0
    nnz = 0
    nreps = 0
    while nnz < N + 1 && nreps < 3
        for mm in mms
            i += 1
            E0, u0[i], Q[i] = qfit(mm, thresh)
            nnz += any(Q[i] .!= 0)
        end
        if nnz < N + 1
            @warn("Insufficent valid points in mismatch2affine. Halving thresh and trying again.")
            thresh /= 2
            nreps += 1
            i = 0
            nnz = 0
        end
    end
    if nreps == 3
        error("Decreased threshold by factor of 8, but it still wasn't enough to avoid degeneracy. It's likely there is a problem with thresh or the mismatch data.")
    end
    # Solve the global sum-over-quadratics problem
    x = Vector{Vector{TFT}}(undef, n)   # knot
    center = convert(Vector{TFT}, ([arraysize(knots)...] .+ 1) / 2)
    for (i, c) in eachknot(knots)
        x[i] = convert(Vector{TFT}, c) - center
    end
    QB = zeros(T, d, d)
    tb = zeros(T, d)
    L = zeros(T, d * (d + 1), d * (d + 1))
    for i in 1:n
        tQ = Q[i]
        tx = x[i]
        xu = tx + u0[i]
        tmp = tQ * xu
        QB += tmp * tx'
        tb += tmp
        for m in 1:d, l in 1:d, k in 1:d, j in 1:d
            L[j + (k - 1) * d, l + (m - 1) * d] += tQ[j, l] * tx[m] * tx[k]
        end
        for l in 1:d, k in 1:d, j in 1:d
            L[j + (k - 1) * d, d * d + l] += tQ[j, l] * tx[k]
        end
        for m in 1:d, l in 1:d, j in 1:d
            L[d * d + j, l + (m - 1) * d] += tQ[j, l] * tx[m]
        end
        for l in 1:d, j in 1:d
            L[d * d + j, d * d + l] += tQ[j, l]
        end
    end
    if all(L .== 0)
        error("All elements of L are zero. It's likely thresh is too high.")
    end
    local rt
    try
        rt = L \ [QB[:];tb]
    catch
        @warn("The data do not suffice to determine a full affine transformation with this grid size---\n  perhaps the only supra-threshold block was the center one?\n  Defaulting to a translation (advice: reconsider your threshold).")
        t = L[(d^2 + 1):end, (d^2 + 1):end] \ tb
        return tformtranslate(convert(Vector{T}, t))
    end
    R = reshape(rt[1:(d * d)], d, d)
    t = rt[(d * d + 1):end]
    return AffineMap(convert(Matrix{T}, R), convert(Vector{T}, t))
end


"""
    u = optimize_per_aperture(mms, thresh)

Compute the naive per-aperture displacement that minimizes the mismatch, treating
each aperture independently. `mms` is a `Vector` of `MismatchArray`s (one per
aperture) and `thresh` is the denominator threshold.

For each aperture, the first shift-dimension component of the `argmin` is
recorded. The returned array has size `(1, length(mms))`.

See also `RegisterCore.argmin_mismatch`.

# Returns
- `u::Matrix{Float64}` of size `(1, n)` — first shift component at the minimum
  for each of the `n` apertures

# Examples
```jldoctest
julia> using RegisterCore

julia> num1 = [(i - 1)^2 + j^2 for i in -5:5, j in -5:5];

julia> num2 = [i^2 + j^2 for i in -5:5, j in -5:5];

julia> mms = [MismatchArray(num1, ones(11, 11)), MismatchArray(num2, ones(11, 11))];

julia> optimize_per_aperture(mms, 0.5)
1×2 Matrix{Float64}:
 1.0  0.0
```
"""
function optimize_per_aperture(mms, thresh)
    gridsize = size(mms)
    nd = length(gridsize)
    u = zeros(nd, gridsize...)
    utmp = zeros(nd)
    for (iblock,mm) in enumerate(mms)
        I = argmin_mismatch(mm, thresh)
        for idim = 1:nd
            u[idim,iblock] = I[idim]
        end
    end
    return u
end


"""
    r = qbuild(E0, umin, Q, maxshift)

Build a `CenterIndexedArray` representing the quadratic mismatch approximation
over the full shift domain `[-maxshift[d], maxshift[d]]`. The quadratic model is:

```
    r[u] = E0 + (u - umin)' * Q * (u - umin)
```

`E0`, `umin`, and `Q` are the outputs of [`qfit`](@ref). Useful for debugging
and visualizing the quadratic fit.

# Returns
- `r::CenterIndexedArray` — evaluated mismatch, indexed from `-maxshift` to `+maxshift`

# Examples
```jldoctest
julia> using RegisterCore

julia> num = [(i - 1)^2 + (j + 2)^2 for i in -5:5, j in -5:5];

julia> mm = MismatchArray(num, ones(11, 11));

julia> E0, umin, Q = qfit(mm, 0.5);

julia> r = qbuild(E0, umin, Q, (5, 5));

julia> r[1, -2]
0.0

julia> r[0, 0]
5.0
```
"""
function qbuild(E0::Real, umin::AbstractVector, Q::AbstractMatrix, maxshift::Union{AbstractVector,Tuple})
    d = length(maxshift)
    (size(Q, 1) == d && size(Q, 2) == d && length(umin) == d) || error("Size mismatch")
    szout = ((2 * [maxshift...] .+ 1)...,)
    out = zeros(eltype(Q), szout)
    j = 1
    du = similar(umin)
    Qdu = similar(umin, typeof(one(eltype(Q)) * one(eltype(du))))
    for c in CartesianIndices(szout)
        for idim in 1:d
            du[idim] = c[idim] - maxshift[idim] - 1 - umin[idim]
        end
        uQu = dot(du, mul!(Qdu, Q, du))
        out[j] = E0 + uQu
        j += 1
    end
    return CenterIndexedArray(out)
end

"""
    tf = uisvalid(u, maxshift)

Return `true` if every entry of the displacement array `u` is within the allowed
domain: `|u[idim, j]| < maxshift[idim] - 0.5001` for all aperture positions `j`
and displacement dimensions `idim`. Returns `false` as soon as any entry violates
this condition.

# Examples
```jldoctest
julia> uisvalid([1.5, 0.5], (3, 3))
true

julia> uisvalid([2.5, 0.5], (3, 3))
false
```
"""
function uisvalid(u::AbstractArray{T}, maxshift::Union{AbstractVector,Tuple}) where {T <: Number}
    nd = size(u, 1)
    sztail = size(u)[2:end]
    for j in CartesianIndices(sztail), idim in 1:nd
        if abs(u[idim, j]) >= maxshift[idim] - register_half
            return false
        end
    end
    return true
end

"""
    uclamp!(u, maxshift)

Clamp the entries of the displacement array `u` in-place so that each satisfies
`|u[idim, j]| ≤ maxshift[idim] - 0.51`. Returns `u`.

Accepts both numeric arrays of shape `(nd, apertures...)` and arrays whose elements
are `StaticVector`s (e.g., `Array{SVector{N,T}}`); both representations are mutated
in-place.

# Examples
```jldoctest
julia> u = [4.0, -5.0];

julia> uclamp!(u, (3, 3))
2-element Vector{Float64}:
  2.49
 -2.49
```
"""
function uclamp!(u::AbstractArray{T}, maxshift::Union{AbstractVector,Tuple}) where {T <: Number}
    nd = size(u, 1)
    sztail = size(u)[2:end]
    for j in CartesianIndices(sztail), idim in 1:nd
        u[idim, j] = max(-maxshift[idim] + register_half_safe, min(u[idim, j], maxshift[idim] - register_half_safe))
    end
    return u
end

function uclamp!(u::AbstractArray{T}, maxshift::Union{AbstractVector,Tuple}) where {T <: StaticVector}
    uclamp!(reshape(reinterpret(eltype(T), vec(u)), (length(T), size(u)...)), maxshift)
    return u
end

"""
    center, cov = principalaxes(img)

Compute the intensity-weighted centroid and covariance of image `img`.
Coordinates are 1-based array indices. `NaN` pixels are ignored.

# Returns
- `center::Vector{T}` — intensity-weighted centroid, length `ndims(img)`
- `cov::Matrix{T}` — `N×N` intensity-weighted covariance matrix

# Examples
```jldoctest
julia> img = zeros(5, 5); img[3, 3] = 1.0;

julia> center, cov = principalaxes(img);

julia> center
2-element Vector{Float64}:
 3.0
 3.0

julia> cov
2×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0
```
"""
function principalaxes(img::AbstractArray{T, N}) where {T, N}
    Ts = typeof(zero(T) / 1)
    psums = pa_init(Ts, size(img))   # partial sums along all but one axis
    # Use a two-pass algorithm
    # First the partial sums, which we use to compute the centroid
    for I in CartesianIndices(axes(img))
        @inbounds v = img[I]
        if !isnan(v)
            @inbounds for d in 1:N
                psums[d][I[d]] += v
            end
        end
    end
    s, m = pa_centroid(psums)
    # Now the variance
    cov = zeros(Ts, N, N)
    for I in CartesianIndices(axes(img))
        @inbounds v = img[I]
        if !isnan(v)
            for j in 1:N
                Δj = I[j] - m[j]
                for i in (j + 1):N
                    cov[i, j] += v * (I[i] - m[i]) * Δj
                end
            end
        end
    end
    for d in 1:N
        cov[d, d] = sum(psums[d] .* ((1:length(psums[d])) .- m[d]) .^ 2)
    end
    for j in 1:N, i in j:N
        cov[i, j] /= s
    end
    for j in 1:N, i in 1:(j - 1)
        cov[i, j] = cov[j, i]
    end
    return m, cov
end

@noinline pa_init(::Type{S}, sz) where {S} = [zeros(S, s) for s in sz]
@noinline function pa_centroid(psums::Vector{Vector{S}}) where {S}
    s = sum(psums[1])
    return s, S[sum(psums[d] .* (1:length(psums[d]))) for d in 1:length(psums)] / s
end

"""
    tfms = pat_rotation(fixed, moving)
    tfms = pat_rotation(fixed, moving, SD)
    tfms = pat_rotation(fixedpa, moving)
    tfms = pat_rotation(fixedpa, moving, SD)

Compute the Principal Axes Transform (PAT) aligning the low-order intensity
moments of two images. `fixed` is the reference image and `moving` is the image
to align. `fixedpa` is a `(center, cov)` tuple from [`principalaxes`](@ref),
useful when aligning many images to the same reference to avoid recomputing its
principal axes.

`SD` is an optional spatial-dimensions matrix that accounts for non-isotropic
sampling (e.g., `SD = Diagonal(voxelspacing)`). Defaults to the identity.

Because intensity ellipsoids are ambiguous up to 180° rotations (sign-flips of
an even number of coordinate axes), the function returns multiple candidate
transforms. Evaluate alignment quality for each candidate and select the best.

# Returns
- `tfms::Vector{AffineMap}` — 2 candidates in 2D, 4 candidates in 3D

# Examples
```jldoctest
julia> fixed = zeros(5, 7); fixed[3, 2:6] .= 1.0;   # horizontal bar

julia> moving = zeros(7, 5); moving[2:6, 3] .= 1.0;  # vertical bar

julia> tfms = pat_rotation(fixed, moving);

julia> length(tfms)
2

julia> tfms[1].linear   # ≈ 90° rotation
2×2 Matrix{Float64}:
  0.0  1.0
 -1.0  0.0
```
"""
function pat_rotation(
        fixedmoments::Tuple{Vector, Matrix}, moving::AbstractArray,
        SD = Matrix{Float64}(I, ndims(moving), ndims(moving))
    )
    nd = ndims(moving)
    nd > 3 && error("Dimensions higher than 3 not supported") # list-generation doesn't yet generalize
    function eigensort2D(var)
        ed = eigen(var)
        if ed.values[1] > ed.values[2]
            return [ed.values[2], ed.values[1]], ed.vectors * [0.0 1.0; 1.0 0.0]
        end
        return ed.values, ed.vectors
    end
    fmean, fvar = fixedmoments
    nd = length(fmean)
    fvar = SD * fvar * SD'
    fD, fV = eigensort2D(fvar)
    mmean, mvar = principalaxes(moving)
    mvar = SD * mvar * SD'
    mD, mV = eigensort2D(mvar)
    R = mV / fV
    if det(R) < 0     # ensure it's a rotation
        R[:, 1] = -R[:, 1]
    end
    c = ([size(moving)...] .+ 1) / 2
    tfms = [pat_at(R, SD, c, fmean, mmean)]
    for i in 1:nd
        for j in (i + 1):nd
            Rc = copy(R)
            Rc[:, i] = -Rc[:, i]
            Rc[:, j] = -Rc[:, j]
            push!(tfms, pat_at(Rc, SD, c, fmean, mmean))
        end
    end
    #     # Debugging check
    #     @show fvar
    #     for i = 1:length(tfms)
    #         Sp = tfms[i].scalefwd
    #         S = SD*Sp/SD
    #         @show S
    #         @show R
    #         @show S'*mvar*S
    #     end
    return tfms
end

pat_rotation(fixed::AbstractArray, moving::AbstractArray, SD = Matrix{Float64}(I, ndims(fixed), ndims(fixed))) =
    pat_rotation(principalaxes(fixed), moving, SD)

function pat_at(S, SD, c, fmean, mmean)
    Sp = SD \ (S * SD)
    bp = (mmean - c) - Sp * (fmean - c)
    return AffineMap(Sp, bp)
end

#### Low-level utilities

@generated function qfit_core!(dE::Array{T, 2}, V4::Array{T, 2}, C::Array{T, 4}, mm::Array{NumDenom{T}, N}, thresh, umin::NTuple{N, Int}, E0, maxsep::NTuple{N, Int}) where {T, N}
    # The cost of generic matrix-multiplies is too high, so we write
    # these out by hand.
    return quote
        @nexprs $N i -> (@nexprs $N j -> j < i ? nothing : (dE_i_j = zero(T); V4_i_j = 0))
        @nexprs $N d -> (umin_d = umin[d])
        @nexprs $N iter1 -> (@nexprs $N iter2 -> iter2 < iter1 ? nothing : (@nexprs $N iter3 -> iter3 < iter2 ? nothing : (@nexprs $N iter4 -> iter4 < iter3 ? nothing : (C_iter1_iter2_iter3_iter4 = zero(T)))))
        @nloops $N i mm begin
            @nif $(N + 1) d -> (abs(i_d - umin[d]) > maxsep[d]) d -> (continue) d -> nothing
            nd = @nref $N mm i
            num, den = nd.num, nd.denom
            if den > thresh
                @nexprs $N d -> (v_d = i_d - umin_d)
                v2 = 0
                @nexprs $N d -> (v2 += v_d * v_d)
                r = num / den
                dE0 = r - E0
                @nexprs $N j -> (@nexprs $N k -> k < j ? nothing : (dE_j_k += dE0 * v_j * v_k; V4_j_k += v2 * v_j * v_k))
                @nexprs $N iter1 -> (@nexprs $N iter2 -> iter2 < iter1 ? nothing : (@nexprs $N iter3 -> iter3 < iter2 ? nothing : (@nexprs $N iter4 -> iter4 < iter3 ? nothing : (C_iter1_iter2_iter3_iter4 += v_iter1 * v_iter2 * v_iter3 * v_iter4))))
            end
        end
        @nexprs $N i -> (@nexprs $N j -> j < i ? (dE[i, j] = dE_j_i; V4[i, j] = V4_j_i) : (dE[i, j] = dE_i_j; V4[i, j] = V4_i_j))
        @nexprs $N iter1 -> (@nexprs $N iter2 -> iter2 < iter1 ? nothing : (@nexprs $N iter3 -> iter3 < iter2 ? nothing : (@nexprs $N iter4 -> iter4 < iter3 ? nothing : (C[iter1, iter2, iter3, iter4] = C_iter1_iter2_iter3_iter4))))
        sortindex = Vector{Int}(undef, 4)
        for iter1 in 1:$N, iter2 in 1:$N, iter3 in 1:$N, iter4 in 1:$N
            sortindex[1] = iter1
            sortindex[2] = iter2
            sortindex[3] = iter3
            sortindex[4] = iter4
            sort!(sortindex)
            C[iter1, iter2, iter3, iter4] = C[sortindex[1], sortindex[2], sortindex[3], sortindex[4]]
        end
        dE, V4, C
    end
end

"""
    E0, u0, Q = qfit(mm, thresh; maxsep=size(mm), opt=true)

Perform a quadratic fit of the mismatch data in `mm`. Returns the mismatch value
`E0` and shift `u0` at the minimum, plus the curvature matrix `Q` of the
best-fit model:

```
    mm ≈ E0 + (u - u0)' * Q * (u - u0)
```

Only shift-locations where `mm[i].denom > thresh` are used. If no valid locations
exist, returns `(zero(T), zeros(T, d), zeros(T, d, d))`.

`maxsep` restricts the fit to shifts satisfying `|u[d] - u0[d]| ≤ maxsep[d]`.
Setting `opt=false` uses a fast heuristic for `Q` instead of a full nonlinear
solve, trading accuracy for speed.

# Returns
- `E0::T` — mismatch value at the fitted minimum
- `u0::Vector{T}` — shift coordinates of the fitted minimum (length `ndims(mm)`)
- `Q::Matrix{T}` — symmetric positive-semidefinite curvature matrix of size `(d, d)`

# Examples
```jldoctest
julia> using RegisterCore

julia> num = [(i - 1)^2 + (j + 2)^2 for i in -5:5, j in -5:5];

julia> mm = MismatchArray(num, ones(11, 11));

julia> E0, u0, Q = qfit(mm, 0.5);

julia> E0
0.0

julia> u0
2-element Vector{Float64}:
  1.0
 -2.0

julia> Q ≈ [1.0 0.0; 0.0 1.0]
true
```
"""
function qfit(mm::MismatchArray, thresh::Real; maxsep = size(mm), opt::Bool = true)
    return qfit(mm, thresh, maxsep, opt)
end

function qfit(mm::MismatchArray, thresh::Real, maxsep, opt::Bool)
    T = eltype(eltype(mm))
    threshT = convert(T, thresh)
    d = ndims(mm)
    mxs = maxshift(mm)
    E0 = typemax(T)
    imin = 0
    for (i, nd) in enumerate(mm)
        if nd.denom > thresh
            r = nd.num / nd.denom
            if r < E0
                imin = i
                E0 = r
            end
        end
    end
    if imin == 0
        return zero(T), zeros(T, d), zeros(T, d, d)  # no valid values
    end
    umin = CartesianIndices(size(mm))[imin]  # not yet relative to center
    uout = T[Tuple(umin)...]
    for i in 1:d
        uout[i] -= (size(mm, i) + 1) >> 1
    end
    dE = Matrix{T}(undef, d, d)
    V4 = similar(dE)
    C = zeros(T, d, d, d, d)
    qfit_core!(dE, V4, C, mm.data, thresh, Tuple(umin), E0, maxsep)
    if all(dE .== 0) || any(diag(V4) .== 0)
        return E0, uout, zeros(eltype(dE), d, d)
    end
    # Initial guess for Q
    M = real(sqrt(V4))::Matrix{T}
    # Compute M\dE/M carefully:
    U, s, V = svd(M)
    sinv = sv_inv(T, s)
    Minv = V * Diagonal(sinv) * U'
    Q = Minv * dE * Minv
    opt || return E0, uout, Q
    local QL
    try
        QL = convert(Matrix{T}, adjoint((cholesky(Hermitian(Q))).U))
    catch err
        if isa(err, LinAlg.PosDefException)
            @warn("Fixing positive-definite exception:")
            @show V4 dE M Q
            QL = convert(Matrix{T}, chol(Q + T(0.001) * mean(diag(Q)) * I, Val{:L}))::Matrix{T}
        else
            rethrow(err)
        end
    end

    # Optimize QL
    x = zeros(T, (d * (d + 1)) >> 1)
    indx = 0
    for i in 1:d,j in 1:d
        if i >= j
            x[indx += 1] = QL[i, j]
        end
    end
    local results
    function solveql(C, dE, QL, x)
        return nlsolve((fx, x) -> QLerr!(x, fx, C, dE, similar(QL)), (gx, x) -> QLjac!(x, gx, C, similar(QL)), x)
    end
    try
        results = solveql(C, dE, QL, x)
    catch err
        @show C dE QL x
        rethrow(err)
    end
    unpackL!(QL, results.zero)
    return E0, uout, QL' * QL
end

@noinline function sv_inv(::Type{T}, s) where {T}
    s1 = s[1]
    return sinv = T[v < sqrt(eps(T)) * s1 ? zero(T) : 1 / v for v in s]
end

"""
    cs, Qs, mmis = mms2fit!(mms, thresh)

Compute per-aperture shifts and quadratic-fit matrices for the N-dimensional
array-of-`MismatchArray`s `mms`, using `thresh` as the denominator threshold.
Also prepares `mms` for interpolation, modifying it in-place after extracting
`cs` and `Qs`.

The dimension `N` of the container `mms` must equal the number of spatial
dimensions of each `MismatchArray` element. For example, a 2×3 matrix of 2D
mismatch arrays is valid; a `Vector` of 2D mismatch arrays is not.

# Returns
- `cs`: `Array{SVector{N,T},N}` — per-aperture shift positions
- `Qs`: `Array{SMatrix{N,N,T},N}` — per-aperture quadratic curvature matrices
- `mmis`: array of interpolated `MismatchArray`s, suitable for input to
  `RegisterPenalty.fixed_λ` and `RegisterPenalty.auto_λ`

# Examples
```jldoctest
julia> using RegisterCore

julia> num1 = [(i - 1)^2 + (j + 2)^2 for i in -5:5, j in -5:5];

julia> num2 = [(i + 1)^2 + (j - 1)^2 for i in -5:5, j in -5:5];

julia> denom = ones(11, 11);

julia> mms = reshape([MismatchArray(num1, denom), MismatchArray(num2, denom)], 1, 2);

julia> cs, Qs, mmis = mms2fit!(mms, 0.5);

julia> cs[1, 1] ≈ [1.0, -2.0]
true
```
"""
function mms2fit!(mms::AbstractArray{A, N}, thresh) where {A <: MismatchArray, N}
    T = eltype(eltype(A))
    gridsize = size(mms)
    cs = Array{SVector{N, T}}(undef, gridsize)
    Qs = Array{similar_type(SArray, T, Size(N, N))}(undef, gridsize)
    for i in 1:length(mms)
        _, cs[i], Qs[i] = qfit(mms[i], thresh; opt = false)
    end
    mmis = interpolate_mm!(mms)
    return cs, Qs, mmis
end

function unpackL!(QL, x)
    d = size(QL, 1)
    indx = 0
    for i in 1:d,j in 1:d
        if i >= j
            QL[i, j] = x[indx += 1]
        end
    end
    return QL
end

function QLerr!(x, fx, C, dE, L)
    d = size(L, 1)
    fill!(L, 0)
    unpackL!(L, x)
    indx = 0
    T = typeof(C[1, 1, 1, 1] * L[1, 1] + C[1, 1, 1, 1] * L[1, 1])
    for i in 1:d, j in 1:d
        if i >= j
            tmp = zero(T)
            for l in 1:d,m in 1:d,n in 1:d
                tmp += L[l, m] * L[l, n] * C[i, j, m, n]
            end
            fx[indx += 1] = tmp - dE[i, j]
        end
    end
    return
end

function QLjac!(x, gx, C, L)
    d = size(L, 1)
    fill!(L, 0)
    unpackL!(L, x)
    T = typeof(C[1, 1, 1, 1] * L[1, 1] + C[1, 1, 1, 1] * L[1, 1])
    indx1 = 0
    for i in 1:d, j in 1:d
        if i >= j
            indx1 += 1
            indx2 = 0
            for a in 1:d, b in 1:d
                if a >= b
                    tmp = zero(T)
                    for k in 1:d
                        tmp += (C[i, j, b, k] + C[i, j, k, b]) * L[a, k]
                    end
                    gx[indx1, indx2 += 1] = tmp
                end
            end
        end
    end
    return
end

end
