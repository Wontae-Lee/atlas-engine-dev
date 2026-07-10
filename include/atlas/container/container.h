#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace atlas {

/**
 * @brief A fixed-size, trivially-copyable inline array usable on both host and device.
 *
 * `Container` is Atlas's `std::array` replacement: an aggregate that stores exactly @p N
 * elements of @p T inside the object itself (member `data_`), with no heap allocation and
 * no owning pointer. Because the storage is inline and every accessor is
 * `__host__ __device__`, a `Container` — when @p T is trivially copyable — is itself
 * trivially copyable and may be captured by value into a device lambda, passed to a CUDA
 * kernel, or embedded in a `DeviceVariant` leaf. `std::array` is unusable here because its
 * member functions are host-only under nvcc; this type supplies device-callable
 * equivalents.
 *
 * The type is `constexpr`-friendly and `ATLAS_FORCE_INLINE` throughout so it compiles down
 * to raw register/local-array accesses in device code.
 *
 * @note Element access via `operator[]`, `data()`, `begin()`/`end()`, and the named
 *       accessors `a()`..`d()` performs no bounds checking and is available on the device.
 *       Only `at()` validates the index, and it is host-only because it throws.
 * @note The named accessors `a()`, `b()`, `c()`, `d()` are the primary way `Container` is
 *       used in geometry: `TriangleContainer4` stores a triangle's three vertices as
 *       `a()`/`b()`/`c()` and its face normal as `d()`.
 *
 * @tparam T Element type. Trivial copyability of `Container<T, N>` follows from that of `T`.
 * @tparam N Element count. `N == 0` is permitted; the storage still reserves one slot (see
 *           `data_`) to keep the array declaration well-formed, but `size()` reports 0 and
 *           `empty()` reports true.
 */
template <typename T, std::size_t N>
struct Container final {
public:
    /**
     * @brief Default-constructs and value-initializes every element to `T{}`.
     *
     * Enabled only when @p N is 0 or `T` is default-constructible. For `N == 0` the single
     * reserved storage slot is still value-initialized.
     *
     * @tparam U SFINAE hook defaulting to `T`; do not supply explicitly.
     */
    template <typename U = T, typename = std::enable_if_t<(N == 0) || std::is_default_constructible_v<U>>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Container() noexcept((N == 0) || std::is_nothrow_default_constructible_v<T>)
        : data_ {} { }

    /**
     * @brief Constructs each element from a corresponding forwarded argument.
     *
     * Element `i` is constructed as `T(std::forward<Args>(args)...[i])`. The overload is
     * enabled only when the number of arguments equals @p N exactly, so every slot is
     * initialized and there is no partial/aggregate-style construction.
     *
     * @tparam Args Argument pack; its size must equal @p N (enforced by SFINAE).
     * @param args One initializer per element, forwarded in order.
     */
    template <typename... Args,
              typename = std::enable_if_t<sizeof...(Args) == N>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr explicit Container(Args&&... args) noexcept(
        (std::is_nothrow_constructible_v<T, Args&&> && ...))
        : data_ { T(std::forward<Args>(args))... } { }

    /**
     * @brief Unchecked mutable element access; host- and device-callable.
     * @param i Index in `[0, N)`; values outside this range are undefined behavior.
     * @return Reference to element @p i.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    operator[](const std::size_t i) noexcept {
        return data_[i];
    }

    /**
     * @brief Unchecked const element access; host- and device-callable.
     * @param i Index in `[0, N)`; values outside this range are undefined behavior.
     * @return Const reference to element @p i.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    operator[](const std::size_t i) const noexcept {
        return data_[i];
    }

    /**
     * @brief Bounds-checked mutable element access; host-only.
     *
     * Host-only because it throws on an out-of-range index, which is not permitted in
     * device code.
     *
     * @param i Index in `[0, N)`.
     * @return Reference to element @p i.
     * @throws std::out_of_range when `i >= N`.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr T&
    at(const std::size_t i) {
        if (i >= N) {
            throw std::out_of_range("atlas::Container::at");
        }
        return data_[i];
    }

    /**
     * @brief Bounds-checked const element access; host-only.
     * @param i Index in `[0, N)`.
     * @return Const reference to element @p i.
     * @throws std::out_of_range when `i >= N`.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr const T&
    at(const std::size_t i) const {
        if (i >= N) {
            throw std::out_of_range("atlas::Container::at");
        }
        return data_[i];
    }

    /**
     * @brief Named accessor for the first element (index 0); mutable.
     *
     * Provided as a readable alias for small fixed layouts — for `TriangleContainer4`,
     * `a()` is the first triangle vertex. Enabled only when the container holds at least
     * one element.
     *
     * @tparam M SFINAE hook defaulting to @p N; do not supply explicitly.
     * @return Reference to element 0.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    a() noexcept {
        return data_[0];
    }

    /**
     * @brief Named accessor for the first element (index 0); const.
     * @tparam M SFINAE hook defaulting to @p N; do not supply explicitly.
     * @return Const reference to element 0.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    a() const noexcept {
        return data_[0];
    }

    /**
     * @brief Named accessor for the second element (index 1); mutable.
     *
     * For `TriangleContainer4` this is the second triangle vertex. Enabled only when the
     * container holds at least two elements.
     *
     * @tparam M SFINAE hook defaulting to @p N; do not supply explicitly.
     * @return Reference to element 1.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    b() noexcept {
        return data_[1];
    }

    /**
     * @brief Named accessor for the second element (index 1); const.
     * @tparam M SFINAE hook defaulting to @p N; do not supply explicitly.
     * @return Const reference to element 1.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    b() const noexcept {
        return data_[1];
    }

    /**
     * @brief Named accessor for the third element (index 2); mutable.
     *
     * For `TriangleContainer4` this is the third triangle vertex. Enabled only when the
     * container holds at least three elements.
     *
     * @tparam M SFINAE hook defaulting to @p N; do not supply explicitly.
     * @return Reference to element 2.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    c() noexcept {
        return data_[2];
    }

    /**
     * @brief Named accessor for the third element (index 2); const.
     * @tparam M SFINAE hook defaulting to @p N; do not supply explicitly.
     * @return Const reference to element 2.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    c() const noexcept {
        return data_[2];
    }

    /**
     * @brief Named accessor for the fourth element (index 3); mutable.
     *
     * For `TriangleContainer4` this slot holds the triangle's face normal rather than a
     * vertex. Enabled only when the container holds at least four elements.
     *
     * @tparam M SFINAE hook defaulting to @p N; do not supply explicitly.
     * @return Reference to element 3.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    d() noexcept {
        return data_[3];
    }

    /**
     * @brief Named accessor for the fourth element (index 3); const.
     * @tparam M SFINAE hook defaulting to @p N; do not supply explicitly.
     * @return Const reference to element 3.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    d() const noexcept {
        return data_[3];
    }

    /**
     * @brief Iterator to the first element; host- and device-callable.
     * @return Pointer to element 0 (equal to `end()` when `N == 0`).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    begin() noexcept {
        return data_;
    }

    /**
     * @brief Const iterator to the first element; host- and device-callable.
     * @return Const pointer to element 0 (equal to `end()` when `N == 0`).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    begin() const noexcept {
        return data_;
    }

    /**
     * @brief Past-the-end iterator; host- and device-callable.
     * @return Pointer to one past the last logical element (`data_ + N`).
     * @note Uses the logical size @p N, so for `N == 0` this equals `begin()` even though
     *       one storage slot physically exists.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    end() noexcept {
        return data_ + N;
    }

    /**
     * @brief Const past-the-end iterator; host- and device-callable.
     * @return Const pointer to one past the last logical element (`data_ + N`).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    end() const noexcept {
        return data_ + N;
    }

    /**
     * @brief The logical element count, known at compile time.
     * @return @p N.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr std::size_t
    size() noexcept {
        return N;
    }

    /**
     * @brief Whether the container holds zero logical elements.
     * @return `true` iff `N == 0`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr bool
    empty() noexcept {
        return N == 0;
    }

    /**
     * @brief Raw mutable pointer to the underlying storage; host- and device-callable.
     * @return Pointer to element 0, valid for `[data(), data() + N)`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    data() noexcept {
        return data_;
    }

    /**
     * @brief Raw const pointer to the underlying storage; host- and device-callable.
     * @return Const pointer to element 0, valid for `[data(), data() + N)`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    data() const noexcept {
        return data_;
    }

    /**
     * @brief Copy-assigns @p value into every logical element.
     * @param value Source copied into each of the @p N slots.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    fill(const T& value) noexcept(
        std::is_nothrow_copy_assignable_v<T>) {
        for (std::size_t i = 0; i < N; ++i) {
            data_[i] = value;
        }
    }

    /**
     * @brief Element-wise swaps this container's storage with @p other.
     *
     * Swaps element-by-element via ADL `swap` (falling back to `std::swap`) rather than
     * exchanging whole objects, so it works in device code where `std::swap` on the
     * aggregate may be unavailable.
     *
     * @param other The container to exchange contents with; must share @p T and @p N.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    swap(Container& other) noexcept(
        std::is_nothrow_swappable_v<T>) {
        using std::swap;
        for (std::size_t i = 0; i < N; ++i) {
            swap(data_[i], other.data_[i]);
        }
    }

public:
    /**
     * @brief Inline element storage.
     *
     * Public so the type stays an aggregate and is trivially copyable when `T` is. The
     * bound is `N == 0 ? 1 : N`: a zero-length C array is ill-formed, so an `N == 0`
     * container still reserves one slot, which the logical-size members (`size()`,
     * `end()`) deliberately ignore.
     */
    T data_[N == 0 ? 1 : N];
};

/**
 * @brief Convenience alias for a two-element `Container`.
 * @tparam T Element type.
 */
template <typename T>
using Container2 = Container<T, 2>;

/**
 * @brief Convenience alias for a three-element `Container`.
 * @tparam T Element type.
 */
template <typename T>
using Container3 = Container<T, 3>;

/**
 * @brief Convenience alias for a four-element `Container`.
 * @tparam T Element type.
 */
template <typename T>
using Container4 = Container<T, 4>;

/**
 * @brief One triangle packed as four `Float3` slots: vertices `a()`,`b()`,`c()` + normal `d()`.
 *
 * This is the element type of a `TriangleMesh`'s device/host triangle buffer. Slots 0..2
 * are the triangle's three corner positions and slot 3 is its outward face normal; device
 * kernels read them through the named accessors. Being a `Container` of trivially-copyable
 * `Float3`, it is itself trivially copyable and safe to store in a `DeviceBuffer` and
 * dereference on the device.
 */
using TriangleContainer4 = Container4<Float3>;

}