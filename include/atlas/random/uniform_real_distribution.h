#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/random/uniform_real_distribution.h>
#else
#include <random>
#endif

namespace atlas {

/**
 * @file uniform_real_distribution.h
 * @brief Backend-selected uniform real distribution alias (Thrust on CUDA, std on CPU).
 *
 * @details
 * This header defines `atlas::uniform_real_distribution<T>` as a thin abstraction over
 * a uniform real-valued random distribution, with the concrete implementation selected
 * at compile time:
 *
 * - **CUDA build (`ATLAS_TASKING_CUDA` defined)**:
 *   - Aliases `thrust::uniform_real_distribution<T>`, which is compatible with
 *     Thrust-based host and device random number generation.
 *
 * - **Non-CUDA build**:
 *   - Aliases `std::uniform_real_distribution<T>` from the C++ standard library.
 *
 * The distribution generates floating-point values uniformly distributed over a
 * half-open interval \f$[a, b)\f$ (subject to backend-specific guarantees).
 *
 * @tparam T Floating-point result type (e.g., `float`, `double`).
 *
 * @note
 * - While the interface is intentionally unified, the underlying implementations
 *   (Thrust vs. standard library) are **not guaranteed to produce identical sequences**
 *   for the same seed and parameters.
 * - This alias is typically used together with `atlas::default_random_engine<T>`
 *   to keep random-number code backend-agnostic.
 * - For strict cross-platform reproducibility, a fixed algorithm and explicit
 *   implementation should be introduced instead of relying on backend defaults.
 */

template <typename T>
#ifdef ATLAS_TASKING_CUDA
using uniform_real_distribution = thrust::uniform_real_distribution<T>;
#else
using uniform_real_distribution = std::uniform_real_distribution<T>;
#endif

} // namespace atlas
