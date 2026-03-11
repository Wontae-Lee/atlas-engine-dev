#pragma once
namespace atlas::math {
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>::Matrix() noexcept {
    // Default-construct as a zero matrix.
    // Layout is a flat contiguous array of size R*C.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = T(0);
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>::Matrix(T s) noexcept {
    // Fill-construct: every element becomes the same scalar s.
    // Useful for quick initialization and testing.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = s;
}
template <typename T, std::size_t R, std::size_t C>
template <typename... Args, typename>
Matrix<T, R, C>::Matrix(Args... args) noexcept
    : _data { static_cast<T>(args)... } {
    // Variadic value constructor:
    //   Matrix<float,2,2> m(1,2,3,4);
    // The number of args is constrained by SFINAE in the declaration.
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>::Matrix(std::initializer_list<T> list) noexcept {
    // Initialize from { ... } in row-major flat order.
    // If list is shorter than R*C, remaining entries are zero-filled.
    // If list is longer, extra values are ignored.
    std::size_t i = 0;
    for (const auto& v : list) {
        if (i >= R * C) break;
        _data[i++] = v;
    }
    for (; i < R * C; ++i) _data[i] = T(0);
}
template <typename T, std::size_t R, std::size_t C>
const T&
Matrix<T, R, C>::operator[](std::size_t i) const noexcept {
    // 1D flat indexing (no bounds check):
    //   i in [0, R*C)
    return _data[i];
}
template <typename T, std::size_t R, std::size_t C>
T&
Matrix<T, R, C>::operator[](std::size_t i) noexcept {
    // Mutable 1D flat indexing (no bounds check).
    return _data[i];
}
template <typename T, std::size_t R, std::size_t C>
const T&
Matrix<T, R, C>::operator()(std::size_t r, std::size_t c) const noexcept {
    // 2D element access via computed flat index.
    // index(r,c) encodes the storage layout (commonly row-major).
    return _data[index(r, c)];
}
template <typename T, std::size_t R, std::size_t C>
T&
Matrix<T, R, C>::operator()(std::size_t r, std::size_t c) noexcept {
    // Mutable 2D element access.
    return _data[index(r, c)];
}
template <typename T, std::size_t R, std::size_t C>
const T&
Matrix<T, R, C>::at(std::size_t r, std::size_t c) const noexcept {
    // Alias of operator()(r,c). Kept for API symmetry with containers.
    // (No bounds checks are performed in this implementation.)
    return _data[index(r, c)];
}
template <typename T, std::size_t R, std::size_t C>
T&
Matrix<T, R, C>::at(std::size_t r, std::size_t c) noexcept {
    // Mutable alias of operator()(r,c).
    return _data[index(r, c)];
}
template <typename T, std::size_t R, std::size_t C>
template <typename E>
Matrix<T, R, C>&
Matrix<T, R, C>::operator=(const MatrixExpression<T, E>& expr) noexcept {
    // Expression-template assignment:
    //   Evaluate the lazy expression 'expr' into this concrete matrix.
    //   This avoids temporaries in chained expressions (e.g., A = B + C * D).
    const E& e = expr();
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = e[i];
    return *this;
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::set(T s) noexcept {
    // Fill the matrix with a single scalar value.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = s;
}
template <typename T, std::size_t R, std::size_t C>
template <typename... Args, typename>
void
Matrix<T, R, C>::set_values(Args... args) noexcept {
    // Bulk set from compile-time known argument pack.
    // Uses a temporary array to ensure a simple unrolled copy loop.
    const T tmp[R * C] { static_cast<T>(args)... };
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = tmp[i];
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::set_zero() noexcept {
    // Explicitly set all entries to zero (common hot-path initializer).
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = T(0);
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::add(T v) noexcept {
    // Element-wise scalar add:
    //   A_ij += v
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] += v;
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::sub(T v) noexcept {
    // Element-wise scalar subtract:
    //   A_ij -= v
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] -= v;
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::mul(T v) noexcept {
    // Element-wise scalar multiply:
    //   A_ij *= v
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] *= v;
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::div(T v) noexcept {
    // Element-wise scalar divide:
    //   A_ij /= v
    // Caller must ensure v != 0 for finite arithmetic.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] /= v;
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::add(const Matrix& m) noexcept {
    // Element-wise matrix add (Hadamard sum):
    //   A_ij += M_ij
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] += m._data[i];
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::sub(const Matrix& m) noexcept {
    // Element-wise matrix subtract:
    //   A_ij -= M_ij
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] -= m._data[i];
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::mul(const Matrix& m) noexcept {
    // Element-wise matrix multiply (Hadamard product):
    //   A_ij *= M_ij
    // Note:
    //   This is NOT matrix multiplication.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] *= m._data[i];
}
template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::div(const Matrix& m) noexcept {
    // Element-wise matrix divide:
    //   A_ij /= M_ij
    // Note:
    //   This is NOT solving a linear system; it's component-wise division.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] /= m._data[i];
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator+=(T v) noexcept {
    // In-place scalar add (delegates to add()).
    add(v);
    return *this;
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator-=(T v) noexcept {
    // In-place scalar subtract.
    sub(v);
    return *this;
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator*=(T v) noexcept {
    // In-place scalar multiply.
    mul(v);
    return *this;
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator/=(T v) noexcept {
    // In-place scalar divide.
    div(v);
    return *this;
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator+=(const Matrix& m) noexcept {
    // In-place element-wise add.
    add(m);
    return *this;
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator-=(const Matrix& m) noexcept {
    // In-place element-wise subtract.
    sub(m);
    return *this;
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator*=(const Matrix& m) noexcept {
    // In-place Hadamard multiply (NOT matmul).
    mul(m);
    return *this;
}
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator/=(const Matrix& m) noexcept {
    // In-place element-wise divide.
    div(m);
    return *this;
}
template <typename T, std::size_t R, std::size_t C>
bool
Matrix<T, R, C>::operator==(const Matrix& other) const noexcept {
    // Exact element-wise equality.
    // For floating point matrices, prefer an approximate comparison in higher-level code.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i)
        if (!(_data[i] == other._data[i])) return false;
    return true;
}
template <typename T, std::size_t R, std::size_t C, std::size_t K>
Matrix<T, R, K>
matmul(const Matrix<T, R, C>& a, const Matrix<T, C, K>& b) noexcept {
    // ------------------------------------------------------------
    // Algorithm: Dense matrix multiplication (triple loop / i-j-k GEMM form)
    //
    // Computes:
    //   out = a * b
    // where:
    //   a is R x C
    //   b is C x K
    //   out is R x K
    //
    // Computation:
    //   out[r,k] = Σ_{c=0..C-1} a[r,c] * b[c,k]
    //
    // Notes:
    //   - This is the straightforward O(R*C*K) algorithm.
    //   - The loop ordering is chosen for clarity (r -> k -> c).
    //   - For performance, a blocked/tiling variant can improve cache reuse.
    // ------------------------------------------------------------
    Matrix<T, R, K> out;
    out.set_zero();
    for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t k = 0; k < K; ++k) {
            T acc = T(0);
            for (std::size_t c = 0; c < C; ++c) {
                acc = static_cast<T>(acc + a(r, c) * b(c, k));
            }
            out(r, k) = acc;
        }
    }
    return out;
}
template <typename T, std::size_t R, std::size_t C>
Vector<T, R>
matmul(const Matrix<T, R, C>& a, const Vector<T, C>& x) noexcept {
    // ------------------------------------------------------------
    // Algorithm: Matrix-vector multiplication (dot per row)
    //
    // Computes:
    //   y = a * x
    // where:
    //   a is R x C
    //   x is length C
    //   y is length R
    //
    // Computation:
    //   y[r] = Σ_{c=0..C-1} a[r,c] * x[c]
    //
    // Notes:
    //   - This is O(R*C).
    //   - Equivalent to taking the dot product of each matrix row with x.
    // ------------------------------------------------------------
    Vector<T, R> y;
    y.set_zero();
    for (std::size_t r = 0; r < R; ++r) {
        T acc = T(0);
        for (std::size_t c = 0; c < C; ++c) {
            acc = static_cast<T>(acc + a(r, c) * x[c]);
        }
        y[r] = acc;
    }
    return y;
}
} // namespace atlas::math
