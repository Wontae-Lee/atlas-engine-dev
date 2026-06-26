# Quaternion Class: Theory and Implementation Notes

This document summarizes the mathematical meaning and implementation structure of the `Quaternion<T>` class. The explanation is written with the provided C++ implementation in mind, but the focus is on the role of quaternions in 3D rotation, interpolation, and transformation.

---

## 1. What Is a Quaternion?

A quaternion is a four-component number usually written as:

$$
q = w + xi + yj + zk
$$

or equivalently as:

$$
q = (w, x, y, z)
$$

where:

```text
w : scalar part
x, y, z : vector part
```

The vector part can be written as:

$$
\mathbf{q}_v = (x, y, z)
$$

so a quaternion can also be represented as:

$$
q = (w, \mathbf{q}_v)
$$

In 3D graphics, physics simulation, robotics, and rigid-body dynamics, quaternions are mainly used to represent rotations.

Compared with Euler angles, quaternions have several advantages:

- They avoid gimbal lock.
- They are compact compared with rotation matrices.
- They are efficient for interpolation.
- They are numerically stable when normalized properly.
- They can represent smooth 3D rotations using unit quaternions.

---

## 2. Unit Quaternion for Rotation

A quaternion represents a pure rotation when it has unit length:

$$
\|q\| = 1
$$

The squared length is:

$$
\|q\|^2 = w^2 + x^2 + y^2 + z^2
$$

The length is:

$$
\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2}
$$

In the implementation:

```cpp
T Quaternion<T>::length_squared() const noexcept {
    return w * w + x * x + y * y + z * z;
}

T Quaternion<T>::length() const noexcept {
    return std::sqrt(length_squared());
}
```

A unit quaternion should satisfy:

$$
w^2 + x^2 + y^2 + z^2 = 1
$$

The `normalize()` function rescales the quaternion to unit length:

```cpp
void Quaternion<T>::normalize() noexcept {
    const T len = length();
    if (len > std::numeric_limits<T>::epsilon()) {
        const T inv = T(1) / len;
        w *= inv;
        x *= inv;
        y *= inv;
        z *= inv;
    }
}
```

---

## 3. Identity Quaternion

The identity rotation is represented by:

$$
q_{identity} = (1, 0, 0, 0)
$$

This quaternion applies no rotation.

In the implementation, the default constructor creates the identity quaternion:

```cpp
constexpr Quaternion<T>::Quaternion() noexcept
    : w(T(1))
    , x(T(0))
    , y(T(0))
    , z(T(0)) {
}
```

The `is_identity()` function checks whether the quaternion is close to this identity value:

```cpp
bool Quaternion<T>::is_identity(T eps) const noexcept {
    return std::abs(w - T(1)) < eps &&
           std::abs(x) < eps &&
           std::abs(y) < eps &&
           std::abs(z) < eps;
}
```

---

## 4. Axis-Angle Representation

A 3D rotation can be described by:

```text
1. a rotation axis
2. a rotation angle
```

Let the normalized rotation axis be:

$$
\mathbf{a} = (a_x, a_y, a_z)
$$

and the rotation angle be:

$$
\theta
$$

The corresponding unit quaternion is:

$$
q = \left(\cos\frac{\theta}{2},\ a_x\sin\frac{\theta}{2},\ a_y\sin\frac{\theta}{2},\ a_z\sin\frac{\theta}{2}\right)
$$

In the implementation:

```cpp
Quaternion<T>::Quaternion(const Vector3<T>& axis, T radians) noexcept {
    const T half = radians * T(0.5);
    const T s    = std::sin(half);
    w            = std::cos(half);
    x            = axis.x * s;
    y            = axis.y * s;
    z            = axis.z * s;
}
```

The same operation is also provided as a static factory function:

```cpp
Quaternion<T> Quaternion<T>::from_axis_angle(const Vector3<T>& axis, T radians) noexcept {
    const T half = radians * T(0.5);
    const T s    = std::sin(half);
    return Quaternion(
        std::cos(half),
        axis.x * s,
        axis.y * s,
        axis.z * s);
}
```

### Important Note

The axis should normally be normalized before constructing a rotation quaternion. If the axis is not unit length, the resulting quaternion will not necessarily be a unit quaternion.

A safe usage pattern is:

```cpp
Vector3<T> axis = input_axis.normalized();
Quaternion<T> q = Quaternion<T>::from_axis_angle(axis, angle);
```

---

## 5. Euler-Angle Construction

The implementation provides construction from Euler angles:

```cpp
Quaternion<T>::Quaternion(T rx, T ry, T rz) noexcept {
    *this = from_euler_xyz(rx, ry, rz);
}
```

The static function is:

```cpp
Quaternion<T> Quaternion<T>::from_euler_xyz(T rx, T ry, T rz) noexcept {
    const T hx = rx * T(0.5);
    const T hy = ry * T(0.5);
    const T hz = rz * T(0.5);
    const T cx = std::cos(hx), sx = std::sin(hx);
    const T cy = std::cos(hy), sy = std::sin(hy);
    const T cz = std::cos(hz), sz = std::sin(hz);
    Quaternion qx(cx, sx, 0, 0);
    Quaternion qy(cy, 0, sy, 0);
    Quaternion qz(cz, 0, 0, sz);
    return qz * qy * qx;
}
```

This constructs three elemental rotations:

```text
qx : rotation around the x-axis
qy : rotation around the y-axis
qz : rotation around the z-axis
```

The final quaternion is:

$$
q = q_z q_y q_x
$$

This means the implementation applies the rotations in XYZ order under the convention used by the quaternion multiplication and vector-rotation routine.

### Important Note on Euler Angles

Euler-angle conventions can be ambiguous. The order `XYZ` may mean different things depending on whether rotations are interpreted as intrinsic or extrinsic rotations. Therefore, when documenting or using this function, the multiplication order should be explicitly stated:

> `from_euler_xyz(rx, ry, rz)` returns `qz * qy * qx`.

---

## 6. Quaternion from a Rotation Matrix

The class can construct a quaternion from a 3x3 rotation matrix:

```cpp
Quaternion<T>::Quaternion(const Matrix<T, 3, 3>& m) noexcept {
    *this = from_matrix3x3(m);
}
```

The implementation uses the matrix trace:

$$
tr = m_{00} + m_{11} + m_{22}
$$

If the trace is positive, the quaternion can be computed using a numerically stable branch:

```cpp
const T tr = m.m00 + m.m11 + m.m22;
if (tr > T(0)) {
    const T s = std::sqrt(tr + T(1)) * T(2);
    return Quaternion(
        s * T(0.25),
        (m.m21 - m.m12) / s,
        (m.m02 - m.m20) / s,
        (m.m10 - m.m01) / s);
}
```

If the trace is not positive, the implementation chooses the largest diagonal component to avoid numerical instability.

This branch-based method is common in rotation-matrix-to-quaternion conversion because it avoids division by very small numbers.

---

## 7. Dot Product

The quaternion dot product is:

$$
q_1 \cdot q_2 = w_1w_2 + x_1x_2 + y_1y_2 + z_1z_2
$$

In the implementation:

```cpp
T Quaternion<T>::dot(const Quaternion& q) const noexcept {
    return w * q.w + x * q.x + y * q.y + z * q.z;
}
```

The dot product is especially important in interpolation.

For two unit quaternions, the dot product measures the cosine of the angular distance in quaternion space. It is also used to decide whether one quaternion should be negated before interpolation so that the shortest rotation path is used.

---

## 8. Conjugate and Inverse

The conjugate of a quaternion is:

$$
q^* = (w, -x, -y, -z)
$$

In the implementation:

```cpp
Quaternion<T> Quaternion<T>::conjugate() const noexcept {
    return Quaternion(w, -x, -y, -z);
}
```

The inverse of a quaternion is:

$$
q^{-1} = \frac{q^*}{\|q\|^2}
$$

In the implementation:

```cpp
Quaternion<T> Quaternion<T>::inverse() const noexcept {
    const T ls = length_squared();
    if (ls > std::numeric_limits<T>::epsilon()) {
        const T inv = T(1) / ls;
        return Quaternion(w * inv, -x * inv, -y * inv, -z * inv);
    }
    return Quaternion();
}
```

For a unit quaternion:

$$
q^{-1} = q^*
$$

Therefore, if a quaternion is guaranteed to be normalized, using the conjugate is enough to undo the rotation.

---

## 9. Rotating a Vector

A 3D vector can be embedded into a pure quaternion:

$$
p = (0, v_x, v_y, v_z)
$$

The rotated vector is computed using:

$$
p' = q p q^{-1}
$$

For a unit quaternion, this becomes:

$$
p' = q p q^*
$$

The implementation uses:

```cpp
Vector3<T> Quaternion<T>::rotate(const Vector3<T>& v) const noexcept {
    Quaternion<T> qv(T(0), v.x, v.y, v.z);
    Quaternion<T> inv = conjugate();
    Quaternion<T> res = (*this) * qv * inv;
    return Vector3<T>(res.x, res.y, res.z);
}
```

### Important Note

This implementation uses `conjugate()` rather than `inverse()`.

That is correct only when the quaternion is normalized. If the quaternion is not unit length, the operation may scale the vector incorrectly.

A safer but slightly more expensive version would be:

```cpp
Quaternion<T> inv = inverse();
```

or the user should ensure:

```cpp
q.normalize();
Vector3<T> rotated = q.rotate(v);
```

---

## 10. Quaternion Multiplication

Quaternion multiplication combines rotations.

Given:

$$
q_1 = (w_1, x_1, y_1, z_1)
$$

and:

$$
q_2 = (w_2, x_2, y_2, z_2)
$$

The product is:

$$
\begin{aligned}
q_1 q_2 = (&w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2, \\
& w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2, \\
& w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2, \\
& w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2)
\end{aligned}
$$

In the implementation:

```cpp
Quaternion<T> Quaternion<T>::operator*(const Quaternion& q) const noexcept {
    return Quaternion(
        w * q.w - x * q.x - y * q.y - z * q.z,
        w * q.x + x * q.w + y * q.z - z * q.y,
        w * q.y - x * q.z + y * q.w + z * q.x,
        w * q.z + x * q.y - y * q.x + z * q.w);
}
```

Quaternion multiplication is not commutative:

$$
q_1 q_2 \ne q_2 q_1
$$

Therefore, multiplication order matters when composing rotations.

---

## 11. Quaternion to Rotation Matrix

A unit quaternion can be converted to a 3x3 rotation matrix.

The implementation computes repeated products first:

```cpp
const T xx = x * x, yy = y * y, zz = z * z;
const T xy = x * y, xz = x * z, yz = y * z;
const T wx = w * x, wy = w * y, wz = w * z;
```

and then fills the matrix as:

```cpp
m.set(
    T(1) - T(2) * (yy + zz),
    T(2) * (xy - wz),
    T(2) * (xz + wy),
    T(2) * (xy + wz),
    T(1) - T(2) * (xx + zz),
    T(2) * (yz - wx),
    T(2) * (xz - wy),
    T(2) * (yz + wx),
    T(1) - T(2) * (xx + yy));
```

The corresponding matrix form is:

$$
R(q) =
\begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}
$$

The 4x4 matrix conversion embeds the 3x3 rotation matrix into the upper-left block of a homogeneous transformation matrix:

```cpp
Matrix<T, 4, 4> m;
m.set_identity();
Matrix<T, 3, 3> r = to_matrix3x3();
m.m00 = r.m00;
m.m01 = r.m01;
m.m02 = r.m02;
m.m10 = r.m10;
m.m11 = r.m11;
m.m12 = r.m12;
m.m20 = r.m20;
m.m21 = r.m21;
m.m22 = r.m22;
```

---

## 12. Linear Interpolation, LERP

Linear interpolation between two quaternions is:

$$
\text{lerp}(a, b, t) = (1 - t)a + tb
$$

In the implementation:

```cpp
Quaternion<T> Quaternion<T>::lerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
    return a * (T(1) - t) + b * t;
}
```

LERP is simple and fast, but the result is generally not a unit quaternion. Therefore, LERP does not preserve constant angular velocity.

---

## 13. Normalized Linear Interpolation, NLERP

NLERP performs LERP and then normalizes the result:

$$
\text{nlerp}(a, b, t) = \frac{(1 - t)a + tb}{\|(1 - t)a + tb\|}
$$

In the implementation:

```cpp
Quaternion<T> Quaternion<T>::nlerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
    Quaternion r = lerp(a, b, t);
    r.normalize();
    return r;
}
```

NLERP is often a good practical choice because it is:

- faster than SLERP
- stable for many animation and simulation cases
- guaranteed to return a normalized quaternion if the interpolation result is nonzero

However, NLERP does not guarantee constant angular velocity.

---

## 14. Spherical Linear Interpolation, SLERP

SLERP interpolates along the shortest arc on the unit quaternion hypersphere.

For unit quaternions `a` and `b`, the interpolation is:

$$
\text{slerp}(a, b, t) = \frac{\sin((1 - t)\theta)}{\sin\theta}a + \frac{\sin(t\theta)}{\sin\theta}b
$$

where:

$$
\cos\theta = a \cdot b
$$

The implementation is:

```cpp
Quaternion<T> Quaternion<T>::slerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
    T cos_theta   = a.dot(b);
    Quaternion bb = b;
    if (cos_theta < T(0)) {
        bb        = Quaternion(-b.w, -b.x, -b.y, -b.z);
        cos_theta = -cos_theta;
    }
    if (cos_theta > T(1) - T(1e-6)) {
        return nlerp(a, bb, t);
    }
    const T theta     = std::acos(cos_theta);
    const T sin_theta = std::sin(theta);
    const T w1        = std::sin((T(1) - t) * theta) / sin_theta;
    const T w2        = std::sin(t * theta) / sin_theta;
    return a * w1 + bb * w2;
}
```

### Shortest-Path Correction

Quaternions `q` and `-q` represent the same physical rotation:

$$
q \equiv -q
$$

If the dot product is negative, the implementation flips `b`:

```cpp
if (cos_theta < T(0)) {
    bb        = Quaternion(-b.w, -b.x, -b.y, -b.z);
    cos_theta = -cos_theta;
}
```

This ensures interpolation follows the shorter rotational path.

### Near-Parallel Fallback

If two quaternions are very close, then:

$$
\sin\theta \approx 0
$$

SLERP can become numerically unstable. The implementation falls back to NLERP:

```cpp
if (cos_theta > T(1) - T(1e-6)) {
    return nlerp(a, bb, t);
}
```

---

## 15. Operators

The class implements arithmetic operators for quaternion algebra.

### Addition and Subtraction

```cpp
Quaternion<T> operator+(const Quaternion& q) const noexcept;
Quaternion<T> operator-(const Quaternion& q) const noexcept;
```

These operate component-wise:

$$
q_1 + q_2 = (w_1 + w_2, x_1 + x_2, y_1 + y_2, z_1 + z_2)
$$

$$
q_1 - q_2 = (w_1 - w_2, x_1 - x_2, y_1 - y_2, z_1 - z_2)
$$

### Scalar Multiplication and Division

```cpp
Quaternion<T> operator*(T s) const noexcept;
Quaternion<T> operator/(T s) const noexcept;
```

These scale each component:

$$
qs = (ws, xs, ys, zs)
$$

$$
\frac{q}{s} = \left(\frac{w}{s}, \frac{x}{s}, \frac{y}{s}, \frac{z}{s}\right)
$$

### Quaternion Multiplication

```cpp
Quaternion<T> operator*(const Quaternion& q) const noexcept;
```

This is the Hamilton product and is used to compose rotations.

---

## 16. Equality Comparison

The equality operator compares components using an epsilon threshold:

```cpp
bool Quaternion<T>::operator==(const Quaternion& q) const noexcept {
    return std::abs(w - q.w) < eps &&
           std::abs(x - q.x) < eps &&
           std::abs(y - q.y) < eps &&
           std::abs(z - q.z) < eps;
}
```

This is appropriate for floating-point values because exact comparison is usually too strict.

### Important Note

The variable `eps` must be defined in the class or surrounding scope. If it is not defined, this function will not compile.

A robust implementation could use:

```cpp
static constexpr T eps = std::numeric_limits<T>::epsilon() * T(10);
```

or accept an epsilon as an explicit argument.

---

## 17. Implementation-Specific Notes

### 17.1 Normalization Assumption

Several operations assume the quaternion represents a rotation, which means it should be normalized.

Important functions affected by this assumption:

```text
- rotate()
- to_matrix3x3()
- to_matrix4x4()
- slerp()
```

For safety, normalize quaternions after construction from arbitrary data:

```cpp
Quaternion<T> q = Quaternion<T>::from_axis_angle(axis.normalized(), angle);
q.normalize();
```

### 17.2 Axis-Angle Input Axis

The axis-angle constructor does not normalize the axis internally. Therefore, the caller should provide a normalized axis.

If the axis length is not one, the quaternion length will be affected.

### 17.3 Matrix Conversion Requires a Valid Rotation Matrix

`from_matrix3x3()` assumes the input matrix is a proper rotation matrix. Ideally, the matrix should be orthonormal and have determinant close to one:

$$
R^T R = I
$$

$$
\det(R) = 1
$$

If the matrix contains scaling, shear, or numerical drift, the resulting quaternion may not be a pure rotation.

### 17.4 Quaternion Sign Ambiguity

The quaternions `q` and `-q` represent the same rotation:

$$
q \equiv -q
$$

This matters when interpolating or comparing rotations. The SLERP implementation handles this by flipping the second quaternion when the dot product is negative.

---

## 18. Practical Usage Examples

### 18.1 Rotate a Vector Around an Axis

```cpp
Vector3<float> axis(0.0f, 1.0f, 0.0f);
axis.normalize();

Quaternion<float> q = Quaternion<float>::from_axis_angle(axis, radians);
Vector3<float> rotated = q.rotate(input_vector);
```

### 18.2 Create a Quaternion from Euler Angles

```cpp
Quaternion<float> q = Quaternion<float>::from_euler_xyz(rx, ry, rz);
q.normalize();
```

### 18.3 Interpolate Between Two Rotations

```cpp
Quaternion<float> q = Quaternion<float>::slerp(q0, q1, t);
q.normalize();
```

### 18.4 Convert to a Rotation Matrix

```cpp
Matrix<float, 3, 3> rotation = q.normalized().to_matrix3x3();
```

---

## 19. Summary

The `Quaternion<T>` class provides a compact and practical representation of 3D rotations.

Core features:

```text
- identity quaternion construction
- axis-angle construction
- Euler-angle construction
- matrix-to-quaternion conversion
- quaternion normalization
- conjugate and inverse operations
- vector rotation
- conversion to 3x3 and 4x4 matrices
- LERP, NLERP, and SLERP interpolation
- quaternion arithmetic operators
```

The most important conceptual points are:

```text
1. Unit quaternions represent rotations.
2. Quaternion multiplication composes rotations.
3. Vector rotation is performed by q * v * q^{-1}.
4. For unit quaternions, q^{-1} equals the conjugate q*.
5. q and -q represent the same physical rotation.
6. SLERP gives smooth spherical interpolation between rotations.
```

For the current implementation, the most important practical caution is:

> Functions such as `rotate()` and `to_matrix3x3()` should be used with normalized quaternions. Otherwise, the result may include unintended scaling or numerical error.