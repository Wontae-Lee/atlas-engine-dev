#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <atlas/core/macros.h>
#include <thrust/tuple.h>

namespace atlas {

/**
 * @brief Backend-adapted tuple type alias for CUDA builds.
 *
 * @tparam Ts Element types stored in the tuple.
 *
 * @details
 * In CUDA-enabled builds, Atlas maps its tuple abstraction to
 * `thrust::tuple<Ts...>` so the same tuple interface can be used in both
 * host and device code.
 *
 * This alias allows generic Atlas code to depend on `atlas::tuple` without
 * having to know whether the underlying implementation comes from Thrust
 * or the C++ standard library.
 */
template <typename... Ts>
using tuple = thrust::tuple<Ts...>;

/**
 * @brief Construct an Atlas tuple from a parameter pack of values.
 *
 * @tparam Ts Types of the tuple elements.
 * @param args Values forwarded into the tuple.
 * @return A tuple containing the provided values.
 *
 * @details
 * This is the CUDA-build backend adapter for tuple construction.
 * It delegates to `thrust::make_tuple(...)` and is marked with
 * @ref ATLAS_ALL_DEVICE so it can be used in both host and device contexts.
 *
 * @note
 * The arguments are passed by value in the current interface.
 */
template <typename... Ts>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    tuple<Ts...>
    make_tuple(Ts... args) {
    return thrust::make_tuple(args...);
}

/**
 * @brief Retrieve the element at compile-time index @p I from a tuple.
 *
 * @tparam I Zero-based compile-time tuple element index.
 * @tparam Tuple Tuple type.
 * @param t Tuple object to access.
 * @return The tuple element at index @p I.
 *
 * @details
 * This function is the CUDA-build backend adapter for tuple element access.
 * It delegates to `thrust::get<I>(...)` and is available in both host and
 * device code.
 *
 * @note
 * The current overload accepts the tuple as a const reference and returns
 * the corresponding element using `auto` return type deduction.
 */
template <std::size_t I, typename Tuple>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
get(const Tuple& t) {
    return thrust::get<I>(t);
}

} // namespace atlas

#else

#include <tuple>

namespace atlas {

/**
 * @brief Backend-adapted tuple type alias for non-CUDA builds.
 *
 * @tparam Ts Element types stored in the tuple.
 *
 * @details
 * In non-CUDA builds, Atlas maps its tuple abstraction to
 * `std::tuple<Ts...>`.
 *
 * This keeps the public interface identical across backends while allowing
 * generic Atlas code to use `atlas::tuple` uniformly.
 */
template <typename... Ts>
using tuple = std::tuple<Ts...>;

/**
 * @brief Construct an Atlas tuple from a parameter pack of values.
 *
 * @tparam Ts Types of the tuple elements.
 * @param args Values forwarded into the tuple.
 * @return A tuple containing the provided values.
 *
 * @details
 * This is the non-CUDA backend adapter for tuple construction.
 * It delegates directly to `std::make_tuple(...)`.
 *
 * @note
 * The arguments are passed by value in the current interface.
 */
template <typename... Ts>
inline tuple<Ts...>
make_tuple(Ts... args) {
    return std::make_tuple(args...);
}

/**
 * @brief Retrieve the element at compile-time index @p I from a tuple.
 *
 * @tparam I Zero-based compile-time tuple element index.
 * @tparam Tuple Tuple type.
 * @param t Tuple object to access.
 * @return The tuple element at index @p I.
 *
 * @details
 * This is the non-CUDA backend adapter for tuple element access.
 * It delegates directly to `std::get<I>(...)`.
 *
 * @note
 * The current overload accepts the tuple as a const reference and returns
 * the corresponding element using `auto` return type deduction.
 */
template <std::size_t I, typename Tuple>
inline auto
get(const Tuple& t) {
    return std::get<I>(t);
}

} // namespace atlas
#endif