# RegisterFit.jl

```@docs
RegisterFit.RegisterFit
```

RegisterFit computes affine transformations that minimize image registration
mismatch. It is part of the [HolyLab](https://github.com/HolyLab) image
registration pipeline.

## Installation

RegisterFit is distributed through the
[HolyLabRegistry](https://github.com/HolyLab/HolyLabRegistry). Add the registry
once, then install:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterFit")
```

## Concepts

### Apertures and mismatch data

The registration pipeline divides an image into overlapping sub-regions called
**apertures**. For each aperture, `RegisterMismatch` computes a `MismatchArray`
that records the normalized squared difference between the fixed and moving images
as a function of shift. Each `MismatchArray` element stores a numerator/denominator
pair; only locations where `denom > thresh` are treated as valid.

### Quadratic fitting

Within a single aperture, the mismatch surface is often well approximated by a
quadratic:

```
mm(u) ≈ E0 + (u − u0)ᵀ Q (u − u0)
```

[`qfit`](@ref) finds `E0` (minimum mismatch value), `u0` (shift at the minimum),
and `Q` (curvature matrix) for one aperture. [`mms2fit!`](@ref) does this for an
entire grid of apertures at once.

### Global affine solve

With one quadratic per aperture, [`mismatch2affine`](@ref) assembles a global
linear system whose solution is the affine transformation (rotation + shear +
translation) that best explains all aperture displacements simultaneously. The
result is an `AffineMap` from
[CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl).

### Principal Axes Transform

[`pat_rotation`](@ref) provides a complementary rigid-alignment approach that
matches the intensity-weighted covariance ellipsoids of two images. It is useful
as an initial guess before running the quadratic optimization. Because an ellipsoid
is symmetric, there are 2 candidate rotations in 2D and 4 in 3D; you must evaluate
each against the mismatch data to pick the best one.

## Typical workflow

```julia
using RegisterFit, RegisterCore

# 1. Obtain per-aperture mismatch arrays from RegisterMismatch (not shown)
#    mms :: Matrix{MismatchArray}  (gridsize matches spatial dims)

# 2. Fit quadratics and prepare for interpolation
cs, Qs, mmis = mms2fit!(mms, thresh)

# 3. Solve for the global affine transform
tform = mismatch2affine(mms, thresh, knots)

# 4. (Optional) rigid pre-alignment with PAT
candidates = pat_rotation(fixed, moving)
# … pick best candidate, then refine with mismatch2affine / RegisterDeformation
```

## Quick examples

### Fitting one aperture

```julia
using RegisterFit, RegisterCore

num = [(i - 1)^2 + (j + 2)^2 for i in -5:5, j in -5:5]
mm  = MismatchArray(num, ones(11, 11))

E0, u0, Q = qfit(mm, 0.5)
# E0 ≈ 0.0, u0 ≈ [1.0, -2.0], Q ≈ I
```

### Rigid alignment via Principal Axes Transform

```julia
using RegisterFit

fixed  = zeros(5, 7); fixed[3, 2:6]  .= 1.0   # horizontal bar
moving = zeros(7, 5); moving[2:6, 3] .= 1.0   # vertical bar

tfms = pat_rotation(fixed, moving)   # 2 candidate AffineMap transforms
# Select the candidate with the smallest mismatch
```
