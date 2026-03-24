#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <cstddef>
#include <type_traits>

namespace atlas {

template <typename T, std::size_t N>
struct Container final {
public:
    template <typename U = T, typename = std::enable_if_t<(N == 0) || std::is_default_constructible_v<U>>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Container() noexcept((N == 0) || std::is_nothrow_default_constructible_v<T>);

    template <typename... Args,
              typename = std::enable_if_t<sizeof...(Args) == N>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr explicit Container(Args&&... args) noexcept(
        (std::is_nothrow_constructible_v<T, Args&&> && ...));

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    operator[](std::size_t i) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    operator[](std::size_t i) const noexcept;

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr T&
    at(std::size_t i);

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr const T&
    at(std::size_t i) const;

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    a() noexcept;

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    a() const noexcept;

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    b() noexcept;

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    b() const noexcept;

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    c() noexcept;

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    c() const noexcept;

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    d() noexcept;

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    d() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    begin() noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    begin() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    end() noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    end() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr std::size_t
    size() noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr bool
    empty() noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    data() noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    data() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    fill(const T& value) noexcept(
        std::is_nothrow_copy_assignable_v<T>);

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    swap(Container& other) noexcept(
        std::is_nothrow_swappable_v<T>);

public:
    T data_[N == 0 ? 1 : N];
};

template <typename T>
using Container2 = Container<T, 2>;

template <typename T>
using Container3 = Container<T, 3>;

template <typename T>
using Container4 = Container<T, 4>;

template <typename T>
using TriangleContainer4 = Container4<Vector3<T>>;

}

#include <atlas/container/container.hpp>
