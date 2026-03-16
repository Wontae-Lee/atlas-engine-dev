#pragma once

#include <stdexcept>
#include <utility>

namespace atlas {

/* =========================
 * Constructors
 * ========================= */

template <typename T, std::size_t N>
template <typename U, typename>
constexpr Container<T, N>::Container() noexcept((N == 0) || std::is_nothrow_default_constructible_v<T>)
    // Value-initialize storage; for N == 0 the single dummy slot stays unused.
    : data_ {} { }

template <typename T, std::size_t N>
template <typename... Args, typename>
constexpr Container<T, N>::Container(Args&&... args) noexcept((std::is_nothrow_constructible_v<T, Args&&> && ...))
    // Materialize each element as T so the inline array is initialized uniformly.
    : data_ { T(std::forward<Args>(args))... } { }

/* =========================
 * Element access
 * ========================= */

template <typename T, std::size_t N>
constexpr T&
Container<T, N>::operator[](std::size_t i) noexcept {
    return data_[i];
}

template <typename T, std::size_t N>
constexpr const T&
Container<T, N>::operator[](std::size_t i) const noexcept {
    return data_[i];
}

template <typename T, std::size_t N>
constexpr T&
Container<T, N>::at(std::size_t i) {
    if (i >= N) {
        throw std::out_of_range("atlas::Container::at");
    }
    return data_[i];
}

template <typename T, std::size_t N>
constexpr const T&
Container<T, N>::at(std::size_t i) const {
    if (i >= N) {
        throw std::out_of_range("atlas::Container::at");
    }
    return data_[i];
}

/* =========================
 * Named accessors
 * ========================= */

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::a() noexcept {
    // M is only used to SFINAE-enable this overload when N >= 1.
    (void)M;
    return data_[0];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::a() const noexcept {
    // M is only used to SFINAE-enable this overload when N >= 1.
    (void)M;
    return data_[0];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::b() noexcept {
    // M is only used to SFINAE-enable this overload when N >= 2.
    (void)M;
    return data_[1];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::b() const noexcept {
    // M is only used to SFINAE-enable this overload when N >= 2.
    (void)M;
    return data_[1];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::c() noexcept {
    // M is only used to SFINAE-enable this overload when N >= 3.
    (void)M;
    return data_[2];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::c() const noexcept {
    // M is only used to SFINAE-enable this overload when N >= 3.
    (void)M;
    return data_[2];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::d() noexcept {
    // M is only used to SFINAE-enable this overload when N >= 4.
    (void)M;
    return data_[3];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::d() const noexcept {
    // M is only used to SFINAE-enable this overload when N >= 4.
    (void)M;
    return data_[3];
}

/* =========================
 * Iterators
 * ========================= */

template <typename T, std::size_t N>
constexpr T*
Container<T, N>::begin() noexcept {
    return data_;
}

template <typename T, std::size_t N>
constexpr const T*
Container<T, N>::begin() const noexcept {
    return data_;
}

template <typename T, std::size_t N>
constexpr T*
Container<T, N>::end() noexcept {
    // end() stays equal to begin() for N == 0, so the dummy slot is never exposed logically.
    return data_ + N;
}

template <typename T, std::size_t N>
constexpr const T*
Container<T, N>::end() const noexcept {
    // end() stays equal to begin() for N == 0, so the dummy slot is never exposed logically.
    return data_ + N;
}

/* =========================
 * Capacity
 * ========================= */

template <typename T, std::size_t N>
constexpr std::size_t
Container<T, N>::size() noexcept {
    return N;
}

template <typename T, std::size_t N>
constexpr bool
Container<T, N>::empty() noexcept {
    return N == 0;
}

/* =========================
 * Raw access
 * ========================= */

template <typename T, std::size_t N>
constexpr T*
Container<T, N>::data() noexcept {
    return data_;
}

template <typename T, std::size_t N>
constexpr const T*
Container<T, N>::data() const noexcept {
    return data_;
}

/* =========================
 * Utilities
 * ========================= */

template <typename T, std::size_t N>
constexpr void
Container<T, N>::fill(const T& value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
    // Fixed-size loop keeps host/device code simple and is typically optimized aggressively.
    for (std::size_t i = 0; i < N; ++i) {
        data_[i] = value;
    }
}

template <typename T, std::size_t N>
constexpr void
Container<T, N>::swap(Container& other) noexcept(std::is_nothrow_swappable_v<T>) {
    using std::swap;
    // Swap element-wise to preserve the same behavior across CPU and CUDA-capable builds.
    for (std::size_t i = 0; i < N; ++i) {
        swap(data_[i], other.data_[i]);
    }
}

} // namespace atlas
