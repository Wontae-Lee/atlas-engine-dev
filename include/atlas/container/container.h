#pragma once

/**
 * @file container.h
 * @brief Fixed-size aggregate container with host/device-compatible element access.
 *
 * @details
 * This header defines @ref atlas::Container, a lightweight statically-sized container used
 * throughout Atlas as a backend-friendly alternative to standard fixed-size containers when
 * host/device annotations are required.
 *
 * `Container<T, N>` provides:
 * - constexpr construction,
 * - indexed and bounds-checked access,
 * - named accessors for the first four elements (`a`, `b`, `c`, `d`),
 * - raw pointer and iterator-style traversal,
 * - fixed-capacity utility functions such as @ref fill and @ref swap.
 *
 * The container stores its elements inline and supports `N == 0` as a valid type.
 */
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <cstddef>
#include <type_traits>

namespace atlas {

/**
 * @brief Fixed-size inline container with host/device-friendly API.
 *
 * @details
 * `Container<T, N>` is a small aggregate-like utility for storing exactly `N` values of type `T`.
 * It is intended for low-overhead use in math, geometry, and simulation code where contiguous
 * storage and device portability are required.
 *
 * The interface intentionally mirrors the common subset of `std::array`:
 * - constant size known at compile time,
 * - contiguous storage,
 * - iterator-style access,
 * - indexed lookup,
 * - fill and swap utilities.
 *
 * For convenience, the first four elements can also be accessed through named accessors
 * @ref a, @ref b, @ref c, and @ref d when the size permits.
 *
 * @tparam T Element type stored in the container.
 * @tparam N Number of elements stored inline.
 *
 * @note
 * `N == 0` is supported. Internally, one dummy element is reserved so the type remains valid,
 * while @ref size returns `0` and @ref empty returns `true`.
 */
template <typename T, std::size_t N>
struct Container final {
public:
    /* =========================
     * Constructors
     * ========================= */

    /**
     * @brief Default-construct all elements.
     *
     * @details
     * Participates in overload resolution only when `N == 0` or `T` is default-constructible.
     *
     * @note For zero-sized containers, no logical elements are constructed.
     */
    template <typename U = T, typename = std::enable_if_t<(N == 0) || std::is_default_constructible_v<U>>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Container() noexcept((N == 0) || std::is_nothrow_default_constructible_v<T>);

    /**
     * @brief Construct the container from exactly `N` element arguments.
     *
     * @param args Element values forwarded into the underlying storage.
     *
     * @note This constructor is enabled only when the number of arguments matches `N`.
     */
    template <typename... Args,
              typename = std::enable_if_t<sizeof...(Args) == N>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr explicit Container(Args&&... args) noexcept(
        (std::is_nothrow_constructible_v<T, Args&&> && ...));

    /* =========================
     * Element access
     * ========================= */

    /**
     * @brief Access an element without bounds checking.
     *
     * @param i Zero-based element index.
     * @return Reference to the element at index `i`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    operator[](std::size_t i) noexcept;

    /**
     * @brief Access an element without bounds checking.
     *
     * @param i Zero-based element index.
     * @return Const reference to the element at index `i`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    operator[](std::size_t i) const noexcept;

    /**
     * @brief Access an element with bounds checking.
     *
     * @param i Zero-based element index.
     * @return Reference to the element at index `i`.
     *
     * @warning Accessing an out-of-range index is invalid; see the implementation for the
     * exact assertion or failure behavior on the active backend.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    at(std::size_t i);

    /**
     * @brief Access an element with bounds checking.
     *
     * @param i Zero-based element index.
     * @return Const reference to the element at index `i`.
     *
     * @warning Accessing an out-of-range index is invalid; see the implementation for the
     * exact assertion or failure behavior on the active backend.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    at(std::size_t i) const;

    /* =========================
     * Named accessors
     * ========================= */

    /**
     * @brief Access the first element.
     *
     * @return Reference to element `0`.
     *
     * @note Enabled only when `N >= 1`.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    a() noexcept;

    /**
     * @brief Access the first element.
     *
     * @return Const reference to element `0`.
     *
     * @note Enabled only when `N >= 1`.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    a() const noexcept;

    /**
     * @brief Access the second element.
     *
     * @return Reference to element `1`.
     *
     * @note Enabled only when `N >= 2`.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    b() noexcept;

    /**
     * @brief Access the second element.
     *
     * @return Const reference to element `1`.
     *
     * @note Enabled only when `N >= 2`.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    b() const noexcept;

    /**
     * @brief Access the third element.
     *
     * @return Reference to element `2`.
     *
     * @note Enabled only when `N >= 3`.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    c() noexcept;

    /**
     * @brief Access the third element.
     *
     * @return Const reference to element `2`.
     *
     * @note Enabled only when `N >= 3`.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    c() const noexcept;

    /**
     * @brief Access the fourth element.
     *
     * @return Reference to element `3`.
     *
     * @note Enabled only when `N >= 4`.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    d() noexcept;

    /**
     * @brief Access the fourth element.
     *
     * @return Const reference to element `3`.
     *
     * @note Enabled only when `N >= 4`.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    d() const noexcept;

    /* =========================
     * Iterators
     * ========================= */

    /**
     * @brief Return an iterator to the first element.
     *
     * @return Pointer to the first stored element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    begin() noexcept;

    /**
     * @brief Return an iterator to the first element.
     *
     * @return Const pointer to the first stored element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    begin() const noexcept;

    /**
     * @brief Return an iterator one past the last element.
     *
     * @return Pointer to the end sentinel.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    end() noexcept;

    /**
     * @brief Return an iterator one past the last element.
     *
     * @return Const pointer to the end sentinel.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    end() const noexcept;

    /* =========================
     * Capacity
     * ========================= */

    /**
     * @brief Return the number of logical elements.
     *
     * @return Compile-time container size `N`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr std::size_t
    size() noexcept;

    /**
     * @brief Check whether the container is empty.
     *
     * @return `true` when `N == 0`, otherwise `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr bool
    empty() noexcept;

    /* =========================
     * Raw access
     * ========================= */

    /**
     * @brief Return a pointer to contiguous storage.
     *
     * @return Pointer to the underlying element buffer.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    data() noexcept;

    /**
     * @brief Return a pointer to contiguous storage.
     *
     * @return Const pointer to the underlying element buffer.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    data() const noexcept;

    /* =========================
     * Utilities
     * ========================= */

    /**
     * @brief Assign the same value to every element.
     *
     * @param value Value copied into each element.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    fill(const T& value) noexcept(
        std::is_nothrow_copy_assignable_v<T>);

    /**
     * @brief Exchange contents with another container of the same type.
     *
     * @param other Container whose elements will be swapped with this instance.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    swap(Container& other) noexcept(
        std::is_nothrow_swappable_v<T>);

public:
    /**
     * @brief Inline storage buffer.
     *
     * @details
     * One element is reserved even for zero-sized containers so `Container<T, 0>` remains a
     * valid complete type.
     */
    T data_[N == 0 ? 1 : N];
};

/* =========================
 * Aliases
 * ========================= */

/// @brief Convenience alias for a two-element @ref Container.
template <typename T>
using Container2 = Container<T, 2>;
/// @brief Convenience alias for a three-element @ref Container.
template <typename T>
using Container3 = Container<T, 3>;
/// @brief Convenience alias for a four-element @ref Container.
template <typename T>
using Container4 = Container<T, 4>;
/// @brief Convenience alias for four 3D vectors, commonly used for tetrahedral-like groupings.
template <typename T>
using TriangleContainer4 = Container4<Vector3<T>>;

} // namespace atlas

#include <atlas/container/container.hpp>
