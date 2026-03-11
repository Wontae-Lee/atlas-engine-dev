#pragma once

/**
 * @file tuple.h
 * @brief Backend-agnostic tuple abstraction (Thrust on CUDA, std::tuple on CPU).
 *
 * @details
 * This header defines a small wrapper API around tuple utilities so that the same
 * code can be compiled for both CUDA and non-CUDA builds without modification.
 *
 * Backend selection:
 * - **CUDA build (`ATLAS_TASKING_CUDA` defined)**:
 *   - Uses `thrust::tuple`, `thrust::make_tuple`, and `thrust::get`.
 *   - All functions are marked device-callable.
 *
 * - **Non-CUDA build**:
 *   - Uses `std::tuple`, `std::make_tuple`, and `std::get`.
 *
 * This abstraction is primarily intended for:
 * - Zip iterators
 * - Tuple-based kernel arguments
 * - Generic algorithms that must compile for both host and device backends
 *
 * @note
 * - `atlas::tuple` is a **type alias**, not a wrapper class.
 * - Semantics (layout, ABI, constexpr-ness) follow the underlying backend
 *   implementation.
 * - No attempt is made to enforce binary compatibility between CUDA and non-CUDA
 *   builds.
 */

#ifdef ATLAS_TASKING_CUDA
#include <atlas/core/macros.h>
#include <thrust/tuple.h>

namespace atlas {

/**
 * @brief Backend tuple type alias (CUDA).
 *
 * @tparam Ts Element types.
 */
template <typename... Ts>
using tuple = thrust::tuple<Ts...>;

/**
 * @brief Constructs a tuple from the given arguments (CUDA backend).
 *
 * @tparam Ts Argument types.
 * @param args Values to store in the tuple.
 * @return `thrust::tuple<Ts...>` containing the given values.
 *
 * @note
 * - Callable from both host and device code.
 */
template <typename... Ts>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
tuple<Ts...>
make_tuple(Ts... args) {
    return thrust::make_tuple(args...);
}

/**
 * @brief Retrieves the `I`-th element from a tuple (CUDA backend).
 *
 * @tparam I Zero-based element index.
 * @tparam Tuple Tuple type.
 * @param t Input tuple.
 * @return The `I`-th element of the tuple.
 *
 * @note
 * - Equivalent to "thrust::get<I>(t)".
 * - Callable from both host and device code.
 */
template <std::size_t I, typename Tuple>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
auto
get(const Tuple& t) {
    return thrust::get<I>(t);
}

} // namespace atlas

#else // --------------------------------------------------------
// Non-CUDA backend (std::tuple)
// --------------------------------------------------------

#include <tuple>

namespace atlas {

/**
 * @brief Backend tuple type alias (CPU).
 *
 * @tparam Ts Element types.
 */
template <typename... Ts>
using tuple = std::tuple<Ts...>;

/**
 * @brief Constructs a tuple from the given arguments (CPU backend).
 *
 * @tparam Ts Argument types.
 * @param args Values to store in the tuple.
 * @return `std::tuple<Ts...>` containing the given values.
 */
template <typename... Ts>
inline
tuple<Ts...>
make_tuple(Ts... args) {
    return std::make_tuple(args...);
}

/**
 * @brief Retrieves the `I`-th element from a tuple (CPU backend).
 *
 * @tparam I Zero-based element index.
 * @tparam Tuple Tuple type.
 * @param t Input tuple.
 * @return The `I`-th element of the tuple.
 *
 * @note
 * - Equivalent to "std::get<I>(t)".
 */
template <std::size_t I, typename Tuple>
inline
auto
get(const Tuple& t) {
    return std::get<I>(t);
}

} // namespace atlas
#endif
