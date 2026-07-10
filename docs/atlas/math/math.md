# Math

`math` is the engine's set of small, header-only value types every other module
builds geometry and physics on: the scalar constants and guards, the three
3-vectors (`Bool3`, `Float3`, `Int3`), the `Float3x3` matrix, and the
`Quaternion`. It has no umbrella and no dispatch — it is a flat bag of plain
value types and free functions. Every type is trivially copyable and
constexpr-constructible, and every operation is annotated `ATLAS_ALL_DEVICE`, so
these values live inside a `DeviceBuffer<T>`, get captured by value into a device
lambda, and compute identically on host and device. That is the whole reason the
engine has its own vector/matrix/quaternion instead of pulling in a host-only
library.

Include `<atlas/math/math.h>` to get the entire set; each header below is also
independently includable and pulls only what it needs.

## Files

| File | Role |
|---|---|
| `include/atlas/math/math.h` | Umbrella that aggregates all headers below; adds nothing of its own. |
| `include/atlas/math/constants.h` | Scalar constants (`pi`, `eps`, `far`, `inf`, physical constants) and the scalar guards `isfinite`, `sqrt_nonnegative`, `solve_quadratic`. |
| `include/atlas/math/vector/bool3.h` | `Bool3` aggregate + reductions (`all`, `any`, `none`) and logical operators. |
| `include/atlas/math/vector/float3.h` | `Float3`, the core 3-D value, plus all vector arithmetic, geometry, and framing free functions. |
| `include/atlas/math/vector/int3.h` | `Int3`, the grid-index vector, plus its operators and the `Float3` conversions. |
| `include/atlas/math/matrix/float3x3.h` | `Float3x3` (row-major 3x3), linear algebra, and the `rotate*` transform helpers. |
| `include/atlas/math/quaternion.h` | `Quaternion` (w, x, y, z) for rotations, with matrix/Euler/axis-angle conversion and interpolation. |

## The vector types

The three vectors share a standard-layout, public-field design: `&x` doubles as
a pointer to the length-3 (`Bool3` aside) array, so `data()`, `operator[]`, and
`at()` all address the same contiguous storage. None of the index accessors is
bounds-checked — only `0, 1, 2` are valid, and `at()` is a plain alias of
`operator[]`, not a throwing/​clamping variant despite the name.

### `Float3` — the core 3-D value

`Float3` carries positions, velocities, directions, and AABB bounds. Its members
are the mutating / in-place forms (`add`, `sub`, `mul`, `div`, `normalize`, the
compound-assignment operators); the non-mutating arithmetic, comparison, and
geometry live as free functions beside the class. `dot` and `length_squared` are
accumulated with `std::fma` to cut rounding error.

Member and free-function forms coexist for the common geometry operations, and
they compute the same thing — pick whichever reads better at the call site:

```cpp
Float3 a(1, 2, 3), b(4, 5, 6);
float  d1 = a.dot(b);          float  d2 = atlas::dot(a, b);        // equal
Float3 c1 = a.cross(b);        Float3 c2 = atlas::cross(a, b);      // equal
Float3 r1 = v.reflected(n);    Float3 r2 = atlas::reflected(v, n);  // equal
Float3 p1 = v.projected(n);    Float3 p2 = atlas::projected(v, n);  // equal
```

**Prefer `length_squared()` to `length()`** whenever you only need to compare or
threshold magnitudes: it skips the square root. The codebase leans on this — the
degenerate-guard helpers all threshold on a *squared* length
(`normalized_or(v, fallback, min_length_squared)`, `orthonormal_basis(...,
min_length_squared)`), so a caller passes a squared floor and never pays for a
`sqrt` in the reject path.

Normalization is guarded on both forms: `normalize()` leaves an exactly-zero
vector untouched and `normalized()` returns a copy of a zero vector rather than
producing NaN. For directions that may collapse, `normalized_or(v, fallback,
min_length_squared)` returns `fallback` when the squared length does not strictly
exceed the floor — and because the test is written `!(len2 > threshold)`, a NaN
length also takes the fallback.

Other free helpers worth knowing: component-wise `min`/`max` (with `cmin`/`cmax`
aliases used at AABB call sites), `clamp`, `ceil`, `floor`, `abs`, `isfinite`,
the `xy_*` family (`xy_dot`, `xy_length`, `xy_length_squared`, `xy_normalized_or`
— all ignoring z for planar work), `reject` (the orthogonal complement of
`projected`), and the frame builders `tangential`, `orthonormal_basis`, and
`orthogonal_unit_vector`, which construct an orthonormal basis around a normal and
pick their helper axis to stay well away from a degenerate cross product.

### `Int3` — the grid-index vector

`Int3` is the signed-integer counterpart used for spatial-hash cell coordinates
and grid resolutions. It deliberately omits the float-only operations (no
`length`, no `normalize`) and carries only `+`, `-`, unary `-`, integer scaling,
equality, per-axis `min`/`max`/`clamp`, and the relational operators. Two
conversions bridge to `Float3`:

- `to_vector3(Int3)` widens each component to float.
- `to_vector3i(Float3)` truncates each component **toward zero** with
  `static_cast<int>`, which is *not* `floor`: `-1.9f` becomes `-1`, not `-2`.
  Callers mapping negative positions to grid cells must account for that.

### `Bool3` and the comparison trap

`Bool3` is a bare three-`bool` aggregate (no constructors, brace-initializable on
host or device). The important thing to internalize: **the relational operators on
`Float3` and `Int3` are componentwise and return a `Bool3`, not a `bool`.**

```cpp
Float3 a(1, 5, 3), b(2, 2, 3);
Bool3 lt = a < b;              // { true, false, false } — one bool per axis
if (atlas::all(a < b)) { ... } // reduce to a single bool
if (atlas::any(a > b)) { ... }
```

This is why `all`, `any`, and `none` exist. Writing `if (a < b)` is a mistake:
there is no `Float3`-to-`bool` conversion, so you must reduce the `Bool3`
explicitly. Only `operator==` / `operator!=` on the vectors return a plain `bool`
(exact, componentwise equality). `Bool3` also supports componentwise `&`, `|`, and
`!` for combining masks before the reduction.

## `Float3x3` — the 3x3 matrix

`Float3x3` stores nine floats **row-major** through an anonymous union: the named
elements `m<row><col>` and the flat `_data[9]` alias the same storage, with
`(r, c)` at `_data[r*3 + c]`. Access it by name, by flat `operator[]`, or by
`at(r, c)` / `operator()(r, c)` — none bounds-checked.

Construction: default is the zero matrix; the explicit scalar constructor
`Float3x3(s)` builds a *scaled identity* (`s` on the diagonal), so `Float3x3(1.0f)`
== `identity3x3()`; the nine-argument constructor is row-major; and a host-only
brace-list constructor fills `_data` in order, zero-padding a short list.
`zero3x3()` and `identity3x3()` are the named factories.

Composition and application:

```cpp
Float3x3 c = a * b;        // == a.mul(b); row-by-column, fma-accumulated, NOT commutative
Float3   w = m * v;        // == m.mul(v); v treated as a column vector, row·v per component
```

`determinant`, `trace`, and `transpose`/`transposed` are direct. Inversion comes
in a checked and an unchecked flavor — a recurring pattern in this header:

- `inverse()` / `inversed()` and the free `inverse(m)` are **unchecked**: a
  singular matrix divides by a zero determinant and fills the result with
  inf/NaN.
- `try_inverse(out, eps)` writes `out` only when `|det| > eps` and returns
  success, leaving `out` untouched on failure; `is_invertible(eps)` just tests the
  determinant. Both default `eps` to `std::numeric_limits<float>::epsilon()`.

Linear solves mirror that split: `solved(b)` / free `solve(A, b)` are unchecked
(they invert), while `solve(b, x, eps)` / free `solve(A, b, x, eps)` write `x` only
on success and return whether `A` was invertible.

Three transform helpers write through an out-parameter and are safe to call with
the same `Float3` as input and output (they snapshot the input first):

- `rotate(matrix, input, output)` — `output = matrix * input`.
- `rotate_translate(matrix, input, offset, output)` — `matrix * input + offset`
  (local → world).
- `rotate_subtract(matrix, input, offset, output)` — `matrix * (input - offset)`
  (world → a frame centered at `offset`).

## `Quaternion` — rotations

`Quaternion` stores the real part `w` first, then the vector part `(x, y, z)`, so
`data()` addresses a length-4 `(w, x, y, z)` array. The default value is the
identity rotation `(1, 0, 0, 0)`. A quaternion represents a rotation only when it
is unit length: the rotation and matrix methods assume that, while the arithmetic
operators treat it as a plain 4-component value.

Construction:

- `Quaternion(w, x, y, z)` — explicit components.
- `Quaternion(axis, radians)` / `from_axis_angle(axis, radians)` — rotation about
  a (unit) axis; `q = (cos(θ/2), sin(θ/2)·axis)`.
- `Quaternion(rx, ry, rz)` / `from_euler_xyz(rx, ry, rz)` — intrinsic X-then-Y-then-Z
  Euler angles, composed as `qz * qy * qx` (x applied first).
- `explicit Quaternion(m)` / `from_matrix3x3(m)` — from a rotation matrix via
  Shepperd's method, which branches on the trace and the largest diagonal element
  to keep the divisor away from zero near 180-degree rotations.
- Host-only brace-list constructor in `(w, x, y, z)` order, defaulting a missing
  `w` to 1 (an empty list gives the identity).

Composition and application:

```cpp
Quaternion c = a * b;       // Hamilton product: apply b first, then a; NOT commutative
Float3     w = q.rotate(v); // v' = (w^2 - |u|^2)v + 2(u·v)u + 2w(u×v), assumes unit q
```

`rotate` applies the expanded `q v q⁻¹` formula directly (no intermediate matrix);
`to_matrix3x3()` produces the equivalent rotation matrix, and `q.to_matrix3x3().mul(v)`
matches `q.rotate(v)` for a unit quaternion (this round-trip is tested).

Normalization and inverse follow the same guard-against-degenerate style as
`Float3`: `normalize()` / `normalized()` leave a near-zero quaternion (length at
or below float machine epsilon) unchanged; `conjugate()` negates the vector part
(the inverse rotation for a *unit* quaternion); `inverse()` is `conjugate /
length_squared()` and is correct for non-unit quaternions, returning the identity
for a near-zero input. Interpolation offers `lerp` (unnormalized), `nlerp`
(lerp + renormalize), and `slerp` (constant angular velocity; flips `b` onto the
same hemisphere for the short arc and falls back to `nlerp` within ~1e-6 of
parallel).

Two subtleties: `operator==` is an **epsilon** comparison (within `atlas::eps` per
component), unlike the exact `Float3`/`Float3x3` equality; and neither `==` nor
`is_identity()` accounts for the double cover, so `(-1, 0, 0, 0)` is *not* treated
as the identity even though it is the same rotation. The free `isfinite(q)` tests
all four components.

## `constants.h` — constants and scalar guards

Every value is `inline constexpr float`, shared as one compile-time value across
translation units on host and device.

| Symbol | Value / meaning | Units |
|---|---|---|
| `pi` | 3.14159265… truncated to float | radians |
| `SQRT_TWO` | √2, precomputed so thermal-velocity math avoids a runtime `sqrt` | — |
| `eps` | `1e-6f`, near-zero / equality threshold (e.g. `Quaternion::operator==`, `is_identity`) | — |
| `tol` | `1e-6f`, numerically equal to `eps` but named for convergence-tolerance intent | — |
| `far` | `1e30f`, a finite "effectively infinite" sentinel that keeps later arithmetic finite | scene length |
| `inf` | `std::numeric_limits<float>::infinity()`, a true unbounded value | — |
| `boltzmann_constant` | `1.380649e-23f`, the exact 2019-redefinition value | J/K (SI) |
| `gravity` | `9.80665f`, standard surface gravity | m/s² |

`eps` and `tol` are deliberately distinct names for the same number so a call site
can express intent without coupling the two; `far` is preferred over `inf` where a
finite sentinel is required so that differences and comparisons stay finite rather
than producing NaN. (The identifier `far` collides with a legacy Windows keyword —
noted in the header, kept as the project name.)

The header also carries three scalar guards used throughout the geometry and
physics kernels:

- `isfinite(float)` — a host/device wrapper over `std::isfinite` so device code has
  a name-resolvable finiteness check matching the host result. (`float3.h` and
  `quaternion.h` add `Float3` and `Quaternion` overloads.)
- `sqrt_nonnegative(float)` — `sqrt(v)` for `v > 0`, else `0.0f`. A numerical guard
  for quantities that are mathematically non-negative but may dip below zero from
  rounding (`1 - cos²θ`); feeding that to `std::sqrt` would give NaN. This is what
  keeps `spherical_direction` and the samplers' `sin θ` finite at `|cos θ| ≥ 1`.
- `solve_quadratic(a, b, c, t0, t1)` — solves `a t² + b t + c = 0` with the
  sign-aware (Citardauq) form that dodges catastrophic cancellation, returns the
  roots sorted `t0 ≤ t1`, and reports `false` (leaving `t0`/`t1` untouched) for a
  negative discriminant. Degenerate leading coefficients yield `inf` roots rather
  than a divide-by-zero, keeping the function total.

## `spherical_direction` — lifting polar angles into an axis frame

The keystone helper for cone and scatter sampling lives in `float3.h`:

```cpp
Float3 spherical_direction(const Float3& unit_axis, float cos_theta, float phi);
Float3 spherical_direction(float cos_theta, float phi);   // canonical +z pole
```

The three-argument form takes a **reference axis** (which must be unit length), a
polar angle given by its cosine `cos_theta ∈ [-1, 1]`, and an azimuth `phi` in
radians, and returns the unit direction at that polar angle from the axis and that
azimuth about it. It works by building a tangent frame around `unit_axis` with
`unit_axis.tangential()`, then combining:

```
dir = unit_axis * cos_theta + (tangent * cos(phi) + bitangent * sin(phi)) * sin_theta
```

where `sin_theta = sqrt_nonnegative(1 - cos_theta²)` — so a `cos_theta` at or just
past `±1` stays finite instead of producing NaN. In effect it *lifts* a
`(theta, phi)` polar coordinate out of a canonical frame and into the frame whose
pole is `unit_axis`. The two-argument overload is the `unit_axis == +z`
specialization, returning `(sin θ cos φ, sin θ sin φ, cos θ)` without building a
frame.

This is a load-bearing primitive: the DSMC scatter kernel and the direction
samplers in `sampling` reconstruct their sampled directions through
`spherical_direction`, which is why its axis-framing and its `sqrt_nonnegative`
guard matter for correctness across those modules.

## Why the shape is what it is

- **Plain value types, no umbrella.** Unlike the tagged-union modules (collider,
  source, …), math has nothing to dispatch on. Each type is a concrete, standard-
  layout value; each free function knows exactly what it computes.
- **Everything is `ATLAS_ALL_DEVICE` and trivially copyable.** That combination is
  what lets a `Float3`/`Float3x3`/`Quaternion` sit in a `DeviceBuffer<T>` and be
  captured by value into a device lambda and compute the same result on host and
  device. It is the reason the engine rolls its own instead of using a host-only
  math library. The two host-only exceptions are the `std::initializer_list`
  brace-list constructors (that type is unusable in device code).
- **Guard, don't throw.** Degenerate inputs are handled by returning a sentinel,
  a fallback, or `false` — never an exception (there is no exception machinery on
  the device path). Hence the checked/unchecked pairs (`inverse` vs `try_inverse`,
  `solved` vs `solve`), the zero-length-safe normalizers, `sqrt_nonnegative`, and
  the `1e30f` finite `far`. Callers on the hot path pick the unchecked form when
  they have already proven the input is valid.
- **`fma` throughout.** `dot`, `length_squared`, the matrix multiply, and the
  quaternion product all accumulate with `std::fma` to fuse the multiply-adds and
  limit rounding error.
