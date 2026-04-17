#pragma once

/**
 * @file container.h
 * @brief Declares a fixed-size contiguous container with host/device-friendly accessors.
 */

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <cstddef>
#include <type_traits>

namespace atlas {

/**
 * @brief Fixed-size contiguous container.
 *
 * This type provides a lightweight statically sized storage abstraction similar
 * to a small array. It exposes:
 * - unchecked and checked element access
 * - pointer-based iteration
 * - named component accessors for the first four elements
 * - utility operations such as fill() and swap()
 *
 * The container size is determined entirely by the template argument @p N and
 * never changes during the lifetime of the object.
 *
 * The implementation is designed to be usable in both host and device code
 * where supported by the configured macros.
 *
 * @tparam T Element type stored in the container.
 * @tparam N Number of elements in the container.
 */
template <typename T, std::size_t N>
struct Container final {
public:
    /**
     * @brief Default constructor.
     *
     * Participates in overload resolution only when the element type is default
     * constructible, unless the container size is zero.
     *
     * For zero-sized containers, construction is always permitted.
     *
     * @tparam U Helper template parameter used for SFINAE.
     */
    template <typename U = T, typename = std::enable_if_t<(N == 0) || std::is_default_constructible_v<U>>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Container() noexcept((N == 0) || std::is_nothrow_default_constructible_v<T>);

    /**
     * @brief Constructs the container from exactly @p N arguments.
     *
     * Each argument is forwarded to construct the corresponding element in the
     * underlying fixed-size storage.
     *
     * This constructor participates in overload resolution only when the number
     * of provided arguments is exactly equal to @p N.
     *
     * @tparam Args Element construction argument types.
     * @param args Arguments used to initialize the contained elements.
     */
    template <typename... Args,
              typename = std::enable_if_t<sizeof...(Args) == N>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr explicit Container(Args&&... args) noexcept(
        (std::is_nothrow_constructible_v<T, Args&&> && ...));

    /**
     * @brief Returns unchecked mutable access to the element at the given index.
     *
     * No bounds checking is performed.
     *
     * @param i Element index.
     * @return Reference to the requested element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    operator[](std::size_t i) noexcept;

    /**
     * @brief Returns unchecked read-only access to the element at the given index.
     *
     * No bounds checking is performed.
     *
     * @param i Element index.
     * @return Const reference to the requested element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    operator[](std::size_t i) const noexcept;

    /**
     * @brief Returns checked mutable access to the element at the given index.
     *
     * Throws if @p i is out of range.
     *
     * @param i Element index.
     * @return Reference to the requested element.
     *
     * @throw std::out_of_range Thrown when @p i is greater than or equal to @p N.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr T&
    at(std::size_t i);

    /**
     * @brief Returns checked read-only access to the element at the given index.
     *
     * Throws if @p i is out of range.
     *
     * @param i Element index.
     * @return Const reference to the requested element.
     *
     * @throw std::out_of_range Thrown when @p i is greater than or equal to @p N.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr const T&
    at(std::size_t i) const;

    /**
     * @brief Returns mutable access to the first element.
     *
     * This function is available only when the container has at least one element.
     *
     * @tparam M Helper template parameter used for SFINAE.
     * @return Reference to the first element.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    a() noexcept;

    /**
     * @brief Returns read-only access to the first element.
     *
     * This function is available only when the container has at least one element.
     *
     * @tparam M Helper template parameter used for SFINAE.
     * @return Const reference to the first element.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    a() const noexcept;

    /**
     * @brief Returns mutable access to the second element.
     *
     * This function is available only when the container has at least two elements.
     *
     * @tparam M Helper template parameter used for SFINAE.
     * @return Reference to the second element.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    b() noexcept;

    /**
     * @brief Returns read-only access to the second element.
     *
     * This function is available only when the container has at least two elements.
     *
     * @tparam M Helper template parameter used for SFINAE.
     * @return Const reference to the second element.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    b() const noexcept;

    /**
     * @brief Returns mutable access to the third element.
     *
     * This function is available only when the container has at least three elements.
     *
     * @tparam M Helper template parameter used for SFINAE.
     * @return Reference to the third element.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    c() noexcept;

    /**
     * @brief Returns read-only access to the third element.
     *
     * This function is available only when the container has at least three elements.
     *
     * @tparam M Helper template parameter used for SFINAE.
     * @return Const reference to the third element.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    c() const noexcept;

    /**
     * @brief Returns mutable access to the fourth element.
     *
     * This function is available only when the container has at least four elements.
     *
     * @tparam M Helper template parameter used for SFINAE.
     * @return Reference to the fourth element.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    d() noexcept;

    /**
     * @brief Returns read-only access to the fourth element.
     *
     * This function is available only when the container has at least four elements.
     *
     * @tparam M Helper template parameter used for SFINAE.
     * @return Const reference to the fourth element.
     */
    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    d() const noexcept;

    /**
     * @brief Returns an iterator to the first element.
     *
     * @return Pointer to the first element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    begin() noexcept;

    /**
     * @brief Returns a read-only iterator to the first element.
     *
     * @return Const pointer to the first element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    begin() const noexcept;

    /**
     * @brief Returns an iterator one past the last element.
     *
     * @return Pointer one past the last element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    end() noexcept;

    /**
     * @brief Returns a read-only iterator one past the last element.
     *
     * @return Const pointer one past the last element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    end() const noexcept;

    /**
     * @brief Returns the fixed number of elements in the container.
     *
     * This value is determined entirely by the template argument @p N.
     *
     * @return The compile-time container size.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr std::size_t
    size() noexcept;

    /**
     * @brief Returns whether the container is empty.
     *
     * This is equivalent to checking whether @p N is zero.
     *
     * @return True if the container size is zero, false otherwise.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr bool
    empty() noexcept;

    /**
     * @brief Returns a mutable pointer to the underlying contiguous storage.
     *
     * @return Pointer to the first stored element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    data() noexcept;

    /**
     * @brief Returns a read-only pointer to the underlying contiguous storage.
     *
     * @return Const pointer to the first stored element.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    data() const noexcept;

    /**
     * @brief Assigns the specified value to all elements in the container.
     *
     * @param value Value to assign to every element.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    fill(const T& value) noexcept(
        std::is_nothrow_copy_assignable_v<T>);

    /**
     * @brief Exchanges the contents of this container with another container.
     *
     * Both containers must have the same element type and size.
     *
     * @param other Container to swap with.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    swap(Container& other) noexcept(
        std::is_nothrow_swappable_v<T>);

public:
    /**
     * @brief Underlying fixed-size contiguous storage.
     *
     * For zero-sized containers, one dummy element is still reserved so the
     * type remains well-formed in standard C++.
     */
    T data_[N == 0 ? 1 : N];
};

/**
 * @brief Alias for a 2-element container.
 *
 * @tparam T Element type.
 */
template <typename T>
using Container2 = Container<T, 2>;

/**
 * @brief Alias for a 3-element container.
 *
 * @tparam T Element type.
 */
template <typename T>
using Container3 = Container<T, 3>;

/**
 * @brief Alias for a 4-element container.
 *
 * @tparam T Element type.
 */
template <typename T>
using Container4 = Container<T, 4>;

/**
 * @brief Alias for a 4-element container of 3D vectors.
 *
 * This is intended for storing four triangle vertices or four 3D points in a
 * fixed-size structure.
 *
 * @tparam T Scalar type used by the vector components.
 */
template <typename T>
using TriangleContainer4 = Container4<Vector3<T>>;

} // namespace atlas

#include <atlas/container/container.hpp>