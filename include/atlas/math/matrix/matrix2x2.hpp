#pragma once
namespace atlas::math {
template <typename T>
constexpr Matrix<T, 2, 2>::Matrix() noexcept
    : m00(T(0))
    , m01(T(0))
    , m10(T(0))
    , m11(T(0)) {
    // Default-construct to the 2x2 zero matrix.
    // [ 0 0 ]
    // [ 0 0 ]
}
template <typename T>
constexpr Matrix<T, 2, 2>::Matrix(T s) noexcept
    : m00(s)
    , m01(T(0))
    , m10(T(0))
    , m11(s) {
    // Diagonal constructor: builds s * I.
    // [ s 0 ]
    // [ 0 s ]
}
template <typename T>
constexpr Matrix<T, 2, 2>::Matrix(T a00, T a01, T a10, T a11) noexcept
    : m00(a00)
    , m01(a01)
    , m10(a10)
    , m11(a11) {
    // Explicit element constructor in row-major order:
    // [ a00 a01 ]
    // [ a10 a11 ]
}
template <typename T>
Matrix<T, 2, 2>::Matrix(std::initializer_list<T> list) noexcept {
    // Initialize from { ... } in row-major order.
    // Missing values default to 0; extra values are ignored.
    const T* it = list.begin();
    m00         = (it != list.end()) ? *it++ : T(0);
    m01         = (it != list.end()) ? *it++ : T(0);
    m10         = (it != list.end()) ? *it++ : T(0);
    m11         = (it != list.end()) ? *it++ : T(0);
}
template <typename T>
template <typename Expression>
Matrix<T, 2, 2>::Matrix(const MatrixExpression<T, Expression>& expr) noexcept {
    // Expression-template materialization:
    // Evaluates a lazy matrix expression into this concrete 2x2 storage.
    const Expression& e = expr();
    m00                 = static_cast<T>(e(0, 0));
    m01                 = static_cast<T>(e(0, 1));
    m10                 = static_cast<T>(e(1, 0));
    m11                 = static_cast<T>(e(1, 1));
}
template <typename T>
std::size_t
Matrix<T, 2, 2>::rows() noexcept {
    // Compile-time fixed shape query.
    return 2;
}
template <typename T>
std::size_t
Matrix<T, 2, 2>::cols() noexcept {
    // Compile-time fixed shape query.
    return 2;
}
template <typename T>
std::size_t
Matrix<T, 2, 2>::size() noexcept {
    // Total element count: rows * cols.
    return 4;
}
template <typename T>
const T*
Matrix<T, 2, 2>::data() const noexcept {
    // Returns a pointer to the contiguous storage (m00..m11).
    // Layout in memory is row-major:
    //   [m00, m01, m10, m11]
    return _data;
}
template <typename T>
T*
Matrix<T, 2, 2>::data() noexcept {
    // Mutable pointer to contiguous storage.
    return _data;
}
template <typename T>
const T&
Matrix<T, 2, 2>::operator[](std::size_t i) const noexcept {
    // Flat indexing (no bounds check):
    // i=0->m00, 1->m01, 2->m10, 3->m11.
    return _data[i];
}
template <typename T>
T&
Matrix<T, 2, 2>::operator[](std::size_t i) noexcept {
    // Mutable flat indexing.
    return _data[i];
}
template <typename T>
const T&
Matrix<T, 2, 2>::at(std::size_t r, std::size_t c) const noexcept {
    // 2D access mapped to row-major linear index:
    // idx = r * 2 + c.
    // (No bounds check in this implementation.)
    return _data[r * 2 + c];
}
template <typename T>
T&
Matrix<T, 2, 2>::at(std::size_t r, std::size_t c) noexcept {
    // Mutable 2D access in row-major layout.
    return _data[r * 2 + c];
}
template <typename T>
const T&
Matrix<T, 2, 2>::operator()(std::size_t r, std::size_t c) const noexcept {
    // Operator form of 2D access. Same mapping as at(r,c).
    return _data[r * 2 + c];
}
template <typename T>
T&
Matrix<T, 2, 2>::operator()(std::size_t r, std::size_t c) noexcept {
    // Mutable operator form of 2D access.
    return _data[r * 2 + c];
}
template <typename T>
void
Matrix<T, 2, 2>::set_zero() noexcept {
    // Set all entries to zero.
    m00 = m01 = m10 = m11 = T(0);
}
template <typename T>
void
Matrix<T, 2, 2>::set_identity() noexcept {
    // Set to identity matrix I:
    // [ 1 0 ]
    // [ 0 1 ]
    m00 = T(1);
    m01 = T(0);
    m10 = T(0);
    m11 = T(1);
}
template <typename T>
void
Matrix<T, 2, 2>::set(T a00, T a01, T a10, T a11) noexcept {
    // Overwrite all entries explicitly.
    m00 = a00;
    m01 = a01;
    m10 = a10;
    m11 = a11;
}
template <typename T>
void
Matrix<T, 2, 2>::add(T s) noexcept {
    // Element-wise scalar add:
    // A_ij += s
    m00 += s;
    m01 += s;
    m10 += s;
    m11 += s;
}
template <typename T>
void
Matrix<T, 2, 2>::sub(T s) noexcept {
    // Element-wise scalar subtract:
    // A_ij -= s
    m00 -= s;
    m01 -= s;
    m10 -= s;
    m11 -= s;
}
template <typename T>
void
Matrix<T, 2, 2>::mul(T s) noexcept {
    // Element-wise scalar multiply:
    // A_ij *= s
    m00 *= s;
    m01 *= s;
    m10 *= s;
    m11 *= s;
}
template <typename T>
void
Matrix<T, 2, 2>::div(T s) noexcept {
    // Element-wise scalar divide using a reciprocal:
    // A_ij /= s  ==>  A_ij *= (1/s)
    // Caller must ensure s != 0.
    const T inv = T(1) / s;
    m00 *= inv;
    m01 *= inv;
    m10 *= inv;
    m11 *= inv;
}
template <typename T>
void
Matrix<T, 2, 2>::add(const Matrix& m) noexcept {
    // Element-wise matrix addition (Hadamard sum):
    // A_ij += M_ij
    m00 += m.m00;
    m01 += m.m01;
    m10 += m.m10;
    m11 += m.m11;
}
template <typename T>
void
Matrix<T, 2, 2>::sub(const Matrix& m) noexcept {
    // Element-wise matrix subtraction:
    // A_ij -= M_ij
    m00 -= m.m00;
    m01 -= m.m01;
    m10 -= m.m10;
    m11 -= m.m11;
}
template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator+=(T s) noexcept {
    // In-place scalar add.
    add(s);
    return *this;
}
template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator-=(T s) noexcept {
    // In-place scalar subtract.
    sub(s);
    return *this;
}
template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator*=(T s) noexcept {
    // In-place scalar multiply.
    mul(s);
    return *this;
}
template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator/=(T s) noexcept {
    // In-place scalar divide.
    div(s);
    return *this;
}
template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator+=(const Matrix& m) noexcept {
    // In-place element-wise add.
    add(m);
    return *this;
}
template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator-=(const Matrix& m) noexcept {
    // In-place element-wise subtract.
    sub(m);
    return *this;
}
template <typename T>
bool
Matrix<T, 2, 2>::operator==(const Matrix& other) const noexcept {
    // Exact component-wise equality.
    // For floating point, prefer approximate comparisons at higher level.
    for (int i = 0; i < 4; ++i)
        if (_data[i] != other._data[i]) return false;
    return true;
}
template <typename T>
bool
Matrix<T, 2, 2>::operator!=(const Matrix& other) const noexcept {
    // Logical negation of exact equality.
    return !(*this == other);
}
template <typename T>
T
Matrix<T, 2, 2>::determinant() const noexcept {
    // Determinant of a 2x2:
    // det([a b; c d]) = a*d - b*c
    //
    // For floating point, use FMA to reduce rounding error:
    //   fma(a, d, -b*c)
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(m00, m11, -m01 * m10);
    } else {
        return m00 * m11 - m01 * m10;
    }
}
template <typename T>
T
Matrix<T, 2, 2>::trace() const noexcept {
    // Trace of a 2x2: tr(A) = a00 + a11.
    return m00 + m11;
}
template <typename T>
void
Matrix<T, 2, 2>::transpose() noexcept {
    // In-place transpose for 2x2:
    // swap off-diagonal elements.
    const T t = m01;
    m01       = m10;
    m10       = t;
}
template <typename T>
Matrix<T, 2, 2>
Matrix<T, 2, 2>::transposed() const noexcept {
    // Return a transposed copy:
    // [a b; c d]^T = [a c; b d]
    return Matrix(m00, m10, m01, m11);
}
template <typename T>
void
Matrix<T, 2, 2>::inverse() noexcept {
    // In-place analytic inverse for 2x2:
    // A^{-1} = (1/det) * [ d -b; -c a ]
    //
    // Caller must ensure det != 0 (or accept inf/NaN for floats).
    const T det = determinant();
    const T inv = T(1) / det;
    const T a = m00, b = m01, c = m10, d = m11;
    m00 = d * inv;
    m01 = -b * inv;
    m10 = -c * inv;
    m11 = a * inv;
}
template <typename T>
Matrix<T, 2, 2>
Matrix<T, 2, 2>::inversed() const noexcept {
    // Return an inverted copy (same formula as inverse()).
    const T det = determinant();
    const T inv = T(1) / det;
    return Matrix(m11 * inv,
                  -m01 * inv,
                  -m10 * inv,
                  m00 * inv);
}
template <typename T>
bool
Matrix<T, 2, 2>::try_inverse(Matrix& out, T eps) const noexcept {
    // Safe inverse attempt:
    //   - Compute det
    //   - Reject if |det| <= eps (treat as singular / ill-conditioned)
    //   - Otherwise write analytic inverse into 'out'
    const T det = determinant();
    if (std::abs(det) <= eps) return false;
    const T inv = T(1) / det;
    out.m00     = m11 * inv;
    out.m01     = -m01 * inv;
    out.m10     = -m10 * inv;
    out.m11     = m00 * inv;
    return true;
}
template <typename T>
bool
Matrix<T, 2, 2>::is_invertible(T eps) const noexcept {
    // Quick singularity check based on determinant magnitude.
    return std::abs(determinant()) > eps;
}
template <typename T>
Matrix<T, 2, 2>
Matrix<T, 2, 2>::mul(const Matrix& r) const noexcept {
    // ------------------------------------------------------------
    // Algorithm: 2x2 matrix multiplication (closed form)
    //
    // For:
    //   A = [a00 a01]   R = [r00 r01]
    //       [a10 a11]       [r10 r11]
    //
    // Product:
    //   A*R = [a00*r00 + a01*r10   a00*r01 + a01*r11]
    //         [a10*r00 + a11*r10   a10*r01 + a11*r11]
    //
    // Uses std::fma(x, y, z) where available to reduce rounding error:
    //   fma(a01, r10, a00*r00) == a01*r10 + a00*r00
    // ------------------------------------------------------------
    return Matrix(
        std::fma(m01, r.m10, m00 * r.m00),
        std::fma(m01, r.m11, m00 * r.m01),
        std::fma(m11, r.m10, m10 * r.m00),
        std::fma(m11, r.m11, m10 * r.m01));
}
template <typename T>
Vector<T, 2>
Matrix<T, 2, 2>::mul(const Vector<T, 2>& v) const noexcept {
    // ------------------------------------------------------------
    // Algorithm: 2x2 matrix-vector multiplication (two dot products)
    //
    // y = A * v
    // y0 = a00*v0 + a01*v1
    // y1 = a10*v0 + a11*v1
    //
    // Uses FMA to reduce rounding error for floating point.
    // ------------------------------------------------------------
    return Vector<T, 2>(
        std::fma(m01, v[1], m00 * v[0]),
        std::fma(m11, v[1], m10 * v[0]));
}
template <typename T>
Matrix<T, 2, 2>
identity2x2() noexcept {
    // Helper: return 2x2 identity matrix.
    return Matrix<T, 2, 2>(T(1));
}
template <typename T>
Matrix<T, 2, 2>
zero2x2() noexcept {
    // Helper: return 2x2 zero matrix.
    return Matrix<T, 2, 2>(T(0), T(0), T(0), T(0));
}
template <typename T>
Matrix<T, 2, 2>
transpose(const Matrix<T, 2, 2>& m) noexcept {
    // Free-function transpose wrapper (returns a transposed copy).
    return m.transposed();
}
template <typename T>
T
determinant(const Matrix<T, 2, 2>& m) noexcept {
    // Free-function determinant wrapper.
    return m.determinant();
}
template <typename T>
Matrix<T, 2, 2>
inverse(const Matrix<T, 2, 2>& m) noexcept {
    // Free-function inverse wrapper (returns a new inverted matrix).
    return m.inversed();
}
template <typename T>
Matrix<T, 2, 2>
operator+(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b) {
    // Element-wise addition (not matmul).
    return Matrix<T, 2, 2>(
        a.m00 + b.m00,
        a.m01 + b.m01,
        a.m10 + b.m10,
        a.m11 + b.m11);
}
template <typename T>
Matrix<T, 2, 2>
operator-(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b) {
    // Element-wise subtraction (not matmul).
    return Matrix<T, 2, 2>(
        a.m00 - b.m00,
        a.m01 - b.m01,
        a.m10 - b.m10,
        a.m11 - b.m11);
}
template <typename T>
Matrix<T, 2, 2>
operator*(const Matrix<T, 2, 2>& a, T s) {
    // Scalar scale (uniform multiply of every entry).
    return Matrix<T, 2, 2>(
        a.m00 * s,
        a.m01 * s,
        a.m10 * s,
        a.m11 * s);
}
template <typename T>
Matrix<T, 2, 2>
operator*(T s, const Matrix<T, 2, 2>& a) {
    // Commutative scalar scale convenience.
    return a * s;
}
template <typename T>
Matrix<T, 2, 2>
operator/(const Matrix<T, 2, 2>& a, T s) {
    // Scalar division implemented as multiply by reciprocal.
    // Caller must ensure s != 0.
    const T inv = T(1) / s;
    return Matrix<T, 2, 2>(
        a.m00 * inv,
        a.m01 * inv,
        a.m10 * inv,
        a.m11 * inv);
}
template <typename T>
Matrix<T, 2, 2>
operator*(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b) {
    // Matrix multiplication overload:
    // delegates to the closed-form mul() implementation.
    return a.mul(b);
}
template <typename T>
Vector<T, 2>
operator*(const Matrix<T, 2, 2>& a, const Vector<T, 2>& v) {
    // Matrix-vector multiplication overload:
    // delegates to the closed-form mul() implementation.
    return a.mul(v);
}
template <typename T>
Vector<T, 2>
Matrix<T, 2, 2>::solved(const Vector<T, 2>& b) const noexcept {
    // ------------------------------------------------------------
    // Algorithm: Solve a 2x2 linear system via analytic inverse
    //
    // Solve:
    //   A x = b
    // with:
    //   A = [a b01; c d],  x = [x0 x1]^T,  b = [b0 b1]^T
    //
    // Inverse-based solution:
    //   x = A^{-1} b
    // where:
    //   A^{-1} = (1/det) * [ d -b01; -c a ]
    //
    // Thus:
    //   x0 = ( d*b0 - b01*b1) / det
    //   x1 = (-c*b0 +  a*b1) / det
    //
    // Uses FMA for better numerical behavior in float.
    // Caller must ensure the matrix is invertible (det != 0).
    // ------------------------------------------------------------
    const T det = determinant();
    const T inv = T(1) / det;
    const T a = m00, b01 = m01, c = m10, d = m11;
    const T n0 = std::fma(b[0], d, -b01 * b[1]); // d*b0 - b01*b1
    const T n1 = std::fma(a, b[1], -b[0] * c);   // a*b1 - c*b0
    return Vector<T, 2>(n0 * inv, n1 * inv);
}
template <typename T>
bool
Matrix<T, 2, 2>::solve(const Vector<T, 2>& b, Vector<T, 2>& x, T eps) const noexcept {
    // ------------------------------------------------------------
    // Algorithm: Robust-ish solve with singularity check
    //
    // Same analytic solution as solved(), but:
    //   - checks |det| <= eps and returns false if near-singular
    //   - writes result into output parameter x on success
    //
    // This avoids generating inf/NaN for ill-conditioned systems.
    // ------------------------------------------------------------
    const T det = determinant();
    if (std::abs(det) <= eps) return false;
    const T a = m00, b01 = m01, c = m10, d = m11;
    const T inv = T(1) / det;
    const T n0  = std::fma(b[0], d, -b01 * b[1]);
    const T n1  = std::fma(a, b[1], -b[0] * c);
    x           = Vector<T, 2>(n0 * inv, n1 * inv);
    return true;
}
template <typename T>
bool
solve(const Matrix<T, 2, 2>& A, const Vector<T, 2>& b, Vector<T, 2>& x, T eps) noexcept {
    // Free-function wrapper for the checked solve().
    return A.solve(b, x, eps);
}
template <typename T>
Vector<T, 2>
solve(const Matrix<T, 2, 2>& A, const Vector<T, 2>& b) noexcept {
    // Free-function wrapper for the unchecked solve() returning a vector.
    return A.solved(b);
}
} // namespace atlas::math
