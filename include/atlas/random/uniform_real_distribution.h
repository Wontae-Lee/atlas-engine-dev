#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/random/uniform_real_distribution.h>
#else
#include <random>
#endif

namespace atlas {

/**
 * @brief Backend-adapted floating-point uniform random distribution alias.
 *
 * @tparam T Floating-point result type produced by the distribution.
 *
 * @details
 * This alias provides a single distribution name that maps to the appropriate
 * backend implementation depending on the active tasking configuration.
 *
 * Backend mapping:
 * - When `ATLAS_TASKING_CUDA` is enabled:
 *   - aliases to `thrust::uniform_real_distribution<T>`
 * - Otherwise:
 *   - aliases to `std::uniform_real_distribution<T>`
 *
 * The intent is to let generic Atlas code write:
 * @code
 * atlas::uniform_real_distribution<float> dist(0.0f, 1.0f);
 * @endcode
 *
 * without caring whether the underlying implementation comes from Thrust
 * or the C++ standard library.
 *
 * @note
 * This alias is intended for real-valued random sampling. The caller is
 * responsible for choosing a suitable floating-point type `T`.
 */
template <typename T>
#ifdef ATLAS_TASKING_CUDA
using uniform_real_distribution = thrust::uniform_real_distribution<T>;
#else
using uniform_real_distribution = std::uniform_real_distribution<T>;
#endif

} // namespace atlas