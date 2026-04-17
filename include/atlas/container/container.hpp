#pragma once

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T, std::size_t N>
template <typename U, typename>
constexpr Container<T, N>::Container() noexcept((N == 0) || std::is_nothrow_default_constructible_v<T>)

    // Value-initialize the underlying fixed-size storage.
    //
    // This ensures:
    // - class types are default-constructed
    // - scalar types are zero-initialized
    //
    // The noexcept condition reflects whether value-initializing all elements
    // can throw. For an empty container, construction is trivially noexcept.
    : data_ {} { }

template <typename T, std::size_t N>
template <typename... Args, typename>
constexpr Container<T, N>::Container(Args&&... args) noexcept((std::is_nothrow_constructible_v<T, Args&&> && ...))

    // Initialize the internal storage from the provided argument pack.
    //
    // Each argument is forwarded into a T object and used to initialize the
    // corresponding element in the fixed-size array.
    //
    // This constructor is typically constrained so it only participates when
    // the number and types of arguments are appropriate for the container size.
    : data_ { T(std::forward<Args>(args))... } { }

template <typename T, std::size_t N>
constexpr T&
Container<T, N>::operator[](std::size_t i) noexcept {
    // Provide unchecked element access.
    //
    // This matches the common container convention:
    // - fast access
    // - no bounds validation
    // - undefined behavior if i >= N
    return data_[i];
}

template <typename T, std::size_t N>
constexpr const T&
Container<T, N>::operator[](std::size_t i) const noexcept {
    // Const-qualified unchecked element access.
    return data_[i];
}

template <typename T, std::size_t N>
constexpr T&
Container<T, N>::at(std::size_t i) {
    // Provide checked element access.
    //
    // Unlike operator[], this function validates the index and throws an
    // exception when the requested position is outside the valid range.
    if (i >= N) {
        throw std::out_of_range("atlas::Container::at");
    }
    return data_[i];
}

template <typename T, std::size_t N>
constexpr const T&
Container<T, N>::at(std::size_t i) const {
    // Const-qualified checked element access.
    if (i >= N) {
        throw std::out_of_range("atlas::Container::at");
    }
    return data_[i];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::a() noexcept {

    // Named accessor for the first component.
    //
    // The extra template parameter M is typically used only to enable this
    // overload conditionally through SFINAE in the declaration.
    // Casting it to void suppresses unused-template warnings in the definition.
    (void)M;
    return data_[0];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::a() const noexcept {

    // Const-qualified named accessor for the first component.
    (void)M;
    return data_[0];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::b() noexcept {

    // Named accessor for the second component.
    //
    // This is commonly useful for vector-like or tuple-like APIs where semantic
    // component access is preferable to raw indexing.
    (void)M;
    return data_[1];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::b() const noexcept {

    // Const-qualified named accessor for the second component.
    (void)M;
    return data_[1];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::c() noexcept {

    // Named accessor for the third component.
    (void)M;
    return data_[2];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::c() const noexcept {

    // Const-qualified named accessor for the third component.
    (void)M;
    return data_[2];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr T&
Container<T, N>::d() noexcept {

    // Named accessor for the fourth component.
    (void)M;
    return data_[3];
}

template <typename T, std::size_t N>
template <std::size_t M, typename>
constexpr const T&
Container<T, N>::d() const noexcept {

    // Const-qualified named accessor for the fourth component.
    (void)M;
    return data_[3];
}

template <typename T, std::size_t N>
constexpr T*
Container<T, N>::begin() noexcept {
    // Return an iterator to the first element.
    //
    // Since the storage is contiguous, the iterator type is simply T*.
    return data_;
}

template <typename T, std::size_t N>
constexpr const T*
Container<T, N>::begin() const noexcept {
    // Const-qualified iterator to the first element.
    return data_;
}

template <typename T, std::size_t N>
constexpr T*
Container<T, N>::end() noexcept {

    // Return an iterator one past the last element.
    //
    // Pointer arithmetic is valid because the storage is contiguous and has
    // exactly N elements.
    return data_ + N;
}

template <typename T, std::size_t N>
constexpr const T*
Container<T, N>::end() const noexcept {

    // Const-qualified iterator one past the last element.
    return data_ + N;
}

template <typename T, std::size_t N>
constexpr std::size_t
Container<T, N>::size() noexcept {
    // Return the fixed number of elements in the container.
    //
    // Unlike dynamic containers, this value is entirely determined by the
    // template argument N and never changes during the object's lifetime.
    return N;
}

template <typename T, std::size_t N>
constexpr bool
Container<T, N>::empty() noexcept {
    // A fixed-size container is empty if and only if its compile-time size is 0.
    return N == 0;
}

template <typename T, std::size_t N>
constexpr T*
Container<T, N>::data() noexcept {
    // Expose the raw pointer to the underlying contiguous storage.
    //
    // This is useful for interop with low-level APIs that operate on plain
    // contiguous memory blocks.
    return data_;
}

template <typename T, std::size_t N>
constexpr const T*
Container<T, N>::data() const noexcept {
    // Const-qualified access to the underlying contiguous storage.
    return data_;
}

template <typename T, std::size_t N>
constexpr void
Container<T, N>::fill(const T& value) noexcept(std::is_nothrow_copy_assignable_v<T>) {

    // Assign the same value to every element in the container.
    //
    // A simple loop is sufficient because the size is fixed and the storage is
    // contiguous. The noexcept condition depends on whether assigning T can throw.
    for (std::size_t i = 0; i < N; ++i) {
        data_[i] = value;
    }
}

template <typename T, std::size_t N>
constexpr void
Container<T, N>::swap(Container& other) noexcept(std::is_nothrow_swappable_v<T>) {
    using std::swap;

    // Exchange elements pairwise with another container of the same type.
    //
    // Since both containers have the same fixed size, swap is implemented as an
    // element-wise operation over the entire storage.
    for (std::size_t i = 0; i < N; ++i) {
        swap(data_[i], other.data_[i]);
    }
}

}