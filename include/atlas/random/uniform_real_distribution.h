#pragma once

#include <thrust/random/uniform_real_distribution.h>

namespace atlas {

/**
 * @brief A uniform continuous distribution over a half-open real interval.
 *
 * Alias for thrust's uniform real distribution so the same drawing code compiles
 * for host and device. Constructed with `[min, max)` bounds and evaluated by
 * passing an `atlas::default_random_engine&`; each call advances the engine.
 * Throughout the sampling helpers this is instantiated with `min = 0, max = 1`
 * to obtain canonical variates `u ∈ [0, 1)` that are then transformed.
 *
 * @tparam T The floating-point result type; defaults to `float`, matching the
 *           single-precision arithmetic used across the device kernels.
 * @see atlas::default_random_engine
 */
template <typename T = float>
using uniform_real_distribution = thrust::uniform_real_distribution<T>;

}