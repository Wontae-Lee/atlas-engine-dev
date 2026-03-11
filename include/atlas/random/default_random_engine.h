#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/random.h>
#else
#include <random>
#endif

namespace atlas {

/**
 * @file default_random_engine.h
 * @brief Backend-selected default random engine alias (Thrust on CUDA, std on CPU).
 *
 * @details
 * This header defines `atlas::default_random_engine<T>` as a thin alias to a
 * backend-provided pseudo-random number generator (PRNG) engine:
 *
 * - **CUDA build (`ATLAS_TASKING_CUDA` defined)**:
 *   - Aliases `thrust::default_random_engine`, which is designed to be usable in
 *     device-capable Thrust workflows.
 *
 * - **Non-CUDA build**:
 *   - Aliases `std::default_random_engine`, the standard library's default PRNG engine.
 *
 * The alias is templated on `T` to match a common pattern across Atlas (type-tagged aliases),
 * even though the selected engine type itself does not depend on `T`.
 *
 * @tparam T A type tag (typically the scalar type used in the computation). Not used directly.
 *
 * @note
 * - The engine choice is intentionally conservative and portable; if you need stronger statistical
 *   properties or reproducibility guarantees across backends, consider introducing an explicit
 *   engine type or a fixed algorithm (e.g., `mt19937`) behind a similar abstraction.
 * - `thrust::default_random_engine` and `std::default_random_engine` may not be bitwise-identical
 *   across platforms/compilers; do not assume identical sequences between CUDA and non-CUDA builds.
 */

template <typename T>
#ifdef ATLAS_TASKING_CUDA
using default_random_engine = thrust::default_random_engine;
#else
using default_random_engine = std::default_random_engine;
#endif

} // namespace atlas
