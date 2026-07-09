#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace atlas {

template <typename T, std::size_t N>
struct Container final {
public:
    template <typename U = T, typename = std::enable_if_t<(N == 0) || std::is_default_constructible_v<U>>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Container() noexcept((N == 0) || std::is_nothrow_default_constructible_v<T>)
        : data_ {} { }

    template <typename... Args,
              typename = std::enable_if_t<sizeof...(Args) == N>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr explicit Container(Args&&... args) noexcept(
        (std::is_nothrow_constructible_v<T, Args&&> && ...))
        : data_ { T(std::forward<Args>(args))... } { }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    operator[](const std::size_t i) noexcept {
        return data_[i];
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    operator[](const std::size_t i) const noexcept {
        return data_[i];
    }

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr T&
    at(const std::size_t i) {
        if (i >= N) {
            throw std::out_of_range("atlas::Container::at");
        }
        return data_[i];
    }

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE constexpr const T&
    at(const std::size_t i) const {
        if (i >= N) {
            throw std::out_of_range("atlas::Container::at");
        }
        return data_[i];
    }

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    a() noexcept {
        return data_[0];
    }

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    a() const noexcept {
        return data_[0];
    }

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    b() noexcept {
        return data_[1];
    }

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    b() const noexcept {
        return data_[1];
    }

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    c() noexcept {
        return data_[2];
    }

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    c() const noexcept {
        return data_[2];
    }

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T&
    d() noexcept {
        return data_[3];
    }

    template <std::size_t M = N, typename = std::enable_if_t<(M >= 4)>>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T&
    d() const noexcept {
        return data_[3];
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    begin() noexcept {
        return data_;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    begin() const noexcept {
        return data_;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    end() noexcept {
        return data_ + N;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    end() const noexcept {
        return data_ + N;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr std::size_t
    size() noexcept {
        return N;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr bool
    empty() noexcept {
        return N == 0;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
    data() noexcept {
        return data_;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
    data() const noexcept {
        return data_;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    fill(const T& value) noexcept(
        std::is_nothrow_copy_assignable_v<T>) {
        for (std::size_t i = 0; i < N; ++i) {
            data_[i] = value;
        }
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr void
    swap(Container& other) noexcept(
        std::is_nothrow_swappable_v<T>) {
        using std::swap;
        for (std::size_t i = 0; i < N; ++i) {
            swap(data_[i], other.data_[i]);
        }
    }

public:
    T data_[N == 0 ? 1 : N];
};

template <typename T>
using Container2 = Container<T, 2>;

template <typename T>
using Container3 = Container<T, 3>;

template <typename T>
using Container4 = Container<T, 4>;

using TriangleContainer4 = Container4<Float3>;

}