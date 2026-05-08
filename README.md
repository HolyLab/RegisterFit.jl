# RegisterFit.jl

[![CI](https://github.com/HolyLab/RegisterFit.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/RegisterFit.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/HolyLab/RegisterFit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterFit.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/RegisterFit.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/RegisterFit.jl/dev)

RegisterFit computes affine transformations that minimize image registration
mismatch. Given per-aperture mismatch data from
[RegisterMismatch](https://github.com/HolyLab/RegisterMismatch.jl), it finds
the best-fit affine transform by fitting each aperture's mismatch to a quadratic
form and solving a global least-squares problem. It is part of the
[HolyLab image registration pipeline](https://github.com/HolyLab).

## Installation

RegisterFit is distributed through the
[HolyLabRegistry](https://github.com/HolyLab/HolyLabRegistry). Add the registry
once, then install the package:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterFit")
```

## Quick start

### Fitting a single aperture's mismatch to a quadratic

```julia
using RegisterFit, RegisterCore

# Synthetic mismatch: parabola centred at shift (1, -2)
num = [(i - 1)^2 + (j + 2)^2 for i in -5:5, j in -5:5]
mm  = MismatchArray(num, ones(11, 11))

E0, u0, Q = qfit(mm, 0.5)
# E0 ≈ 0.0      — mismatch value at the minimum
# u0 ≈ [1, -2]  — shift at the minimum
# Q  ≈ I        — curvature matrix
```

### Rigid alignment via the Principal Axes Transform

```julia
using RegisterFit

# Horizontal bar in fixed, vertical bar in moving (≈ 90° rotation)
fixed  = zeros(5, 7); fixed[3, 2:6]  .= 1.0
moving = zeros(7, 5); moving[2:6, 3] .= 1.0

tfms = pat_rotation(fixed, moving)   # 2 candidate AffineMap transforms in 2D
# Evaluate each candidate against mismatch data and pick the best
```

## Overview

The typical workflow in the registration pipeline is:

1. **Compute mismatch** — per-aperture `MismatchArray`s from `RegisterMismatch`
2. **`mms2fit!`** — fit each aperture to a quadratic; returns per-aperture shifts
   and curvature matrices
3. **`mismatch2affine`** — global least-squares solve over all apertures to
   produce a single `AffineMap`
4. **Refine** — further optimisation in `RegisterDeformation`

See the [documentation](https://HolyLab.github.io/RegisterFit.jl/stable) for
the full API reference.
