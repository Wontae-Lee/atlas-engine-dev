#pragma once

/**
 * @file container.h
 * @brief Declares a fixed-size backend-portable aggregate container.
 *
 * The Container type is a lightweight alternative to std::array that is
 * annotated for Atlas host/device portability. It is used across geometry,
 * math, and helper code where a trivially copyable fixed-size aggregate is
 * preferable to heavier abstractions.
 */

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <cstddef>
#include <type_traits>

namespace atlas {

/**
 * @brief Fixed-size aggregate container with host/device-friendly accessors.
 *
 * Container stores exactly @p N elements of type @p T in contiguous storage.
 * It exposes array-style indexing, checked access through at(), symbolic
 * accessors a()/b()/c()/d() for small tuples, iterator support, and basic
 * utility operations such as fill() and swap().
 *
 * The class is intentionally simple and constexpr-friendly so it can be used
 * in both compile-time and device-executed contexts.
 *
 * @tparam T Element type stored in the container.
 * @tparam N Number of elements.
 */
template <typename T, std::size_t N>
struct Container final {
public:
    /**
     * @brief Default-constructs all elements when possible.
     *
     * This constructor participates in overload resolution only when the
     * element type is default-constructible, except for the degenerate N==0
     * case which is always permitted.
     */
    template <typename U = T, typename = std::enable_if_t<(N == 0) || std::is_default_constructible_v<U>>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Container() noexcept((N == 0) || std::is_nothrow_default_constructible_v<T>);

    /**
     * @brief Constructs the container from exactly @p N values.
     *
     * @tparam Args Argument pack used to initialize the elements.
     * @param args Values forwarded into the fixed storage.
     */
    template <typename... Args,
              typename = std::enable_if_t<sizeof...(Args) == N>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr explicit Container(Args&&... args) noexcept(
        (std::is_nothrow_constructible_v<T, Args&&> && ...));

    /**
     * @brief Returns mutable element access without bounds checking.
     *
     * @param i Element index.
     * @return T& Reference to the indexed element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    operator[](std::size_t i) noexcept;

    /**
     * @brief Returns const element access without bounds checking.
     *
     * @param i Element index.
     * @return const T& Reference to the indexed element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    operator[](std::size_t i) const noexcept;

    /**
     * @brief Returns mutable checked element access.
     *
     * @param i Element index.
     * @return T& Reference to the indexed element.
     * @throws std::out_of_range if @p i is not in [0, N).
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr T&
    at(std::size_t i);

    /**
     * @brief Returns const checked element access.
     *
     * @param i Element index.
     * @return const T& Reference to the indexed element.
     * @throws std::out_of_range if @p i is not in [0, N).
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr const T&
    at(std::size_t i) const;

    /**
     * @brief Returns the first element using tuple-style naming.
     *
     * This accessor is enabled only when the container has at least one entry.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    a() noexcept;

    /**
     * @brief Returns the first element using tuple-style naming.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    a() const noexcept;

    /**
     * @brief Returns the second element using tuple-style naming.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    b() noexcept;

    /**
     * @brief Returns the second element using tuple-style naming.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    b() const noexcept;

    /**
     * @brief Returns the third element using tuple-style naming.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    c() noexcept;

    /**
     * @brief Returns the third element using tuple-style naming.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    c() const noexcept;

    /**
     * @brief Returns the fourth element using tuple-style naming.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    d() noexcept;

    /**
     * @brief Returns the fourth element using tuple-style naming.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    d() const noexcept;

    /**
     * @brief Returns an iterator to the first element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    begin() noexcept;

    /**
     * @brief Returns a const iterator to the first element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    begin() const noexcept;

    /**
     * @brief Returns an iterator one past the last element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    end() noexcept;

    /**
     * @brief Returns a const iterator one past the last element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    end() const noexcept;

    /**
     * @brief Returns the compile-time number of stored elements.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr std::size_t
    size() noexcept;

    /**
     * @brief Returns whether the container has zero logical elements.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr bool
    empty() noexcept;

    /**
     * @brief Returns a mutable pointer to the contiguous storage.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    data() noexcept;

    /**
     * @brief Returns a const pointer to the contiguous storage.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    data() const noexcept;

    /**
     * @brief Assigns the same value to every logical element.
     *
     * @param value Value copied into each slot.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    fill(const T& value) noexcept(
        std::is_nothrow_copy_assignable_v<T>);

    /**
     * @brief Swaps each element with the corresponding element of another container.
     *
     * @param other Container with the same shape and element type.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    swap(Container& other) noexcept(
        std::is_nothrow_swappable_v<T>);

public:
    ///< Underlying contiguous storage. A single dummy slot is preserved for N==0.
    T data_[N == 0 ? 1 : N];
};

/**
 * @brief Convenience alias for a two-element Container.
 */
template <typename T>
using Container2 = Container<T, 2>;

/**
 * @brief Convenience alias for a three-element Container.
 */
template <typename T>
using Container3 = Container<T, 3>;

/**
 * @brief Convenience alias for a four-element Container.
 */
template <typename T>
using Container4 = Container<T, 4>;

/**
 * @brief Fixed-size triangle vertex container used by mesh utilities.
 */
template <typename T>
using TriangleContainer4 = Container4<Vector3<T>>;

}

#include <atlas/container/container.hpp>
