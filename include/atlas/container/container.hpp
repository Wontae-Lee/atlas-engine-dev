#pragma once

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T, std::size_t N>
template <typename U, typename>
constexpr Container<T, N>::Container() noexcept((N == 0) || std::is_nothrow_default_constructible_v<T>)

    // Aggregate-initialize all logical elements.
    : data_ {} { }

template <typename T, std::size_t N>
template <typename... Args, typename>
constexpr Container<T, N>::Container(Args&&... args) noexcept((std::is_nothrow_constructible_v<T, Args&&> && ...))

    // Materialize the fixed-size tuple directly into contiguous storage.
    : data_ { T(std::forward<Args>(args))... } { }

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

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::a() noexcept {

    // M is used only to constrain availability at compile time.
    (void)M;
    return data_[0];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::a() const noexcept {

    // M is used only to constrain availability at compile time.
    (void)M;
    return data_[0];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::b() noexcept {

    // M is used only to constrain availability at compile time.
    (void)M;
    return data_[1];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::b() const noexcept {

    // M is used only to constrain availability at compile time.
    (void)M;
    return data_[1];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::c() noexcept {

    // M is used only to constrain availability at compile time.
    (void)M;
    return data_[2];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::c() const noexcept {

    // M is used only to constrain availability at compile time.
    (void)M;
    return data_[2];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::d() noexcept {

    // M is used only to constrain availability at compile time.
    (void)M;
    return data_[3];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::d() const noexcept {

    // M is used only to constrain availability at compile time.
    (void)M;
    return data_[3];
}

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

    return data_ + N;
}

template <typename T, std::size_t N>
constexpr const T*
Container<T, N>::end() const noexcept {

    return data_ + N;
}

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

template <typename T, std::size_t N>
constexpr void
Container<T, N>::fill(const T& value) noexcept(std::is_nothrow_copy_assignable_v<T>) {

    // Preserve host/device portability with an explicit loop instead of std::fill.
    for (std::size_t i = 0; i < N; ++i) {
        data_[i] = value;
    }
}

template <typename T, std::size_t N>
constexpr void
Container<T, N>::swap(Container& other) noexcept(std::is_nothrow_swappable_v<T>) {
    using std::swap;

    // Swap element-by-element to keep the implementation constexpr-friendly.
    for (std::size_t i = 0; i < N; ++i) {
        swap(data_[i], other.data_[i]);
    }
}

}
