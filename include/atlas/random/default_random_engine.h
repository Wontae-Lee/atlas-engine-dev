#pragma once

#include <thrust/random.h>

namespace atlas {

/**
 * @brief The engine-wide pseudo-random number generator type.
 *
 * A thin alias for thrust's default RNG so that both host and device code draw
 * from a single, consistent engine type. Because it is a thrust engine it is
 * trivially copyable and usable from inside a `__host__ __device__` lambda: each
 * thread typically constructs its own engine, seeds it (often by discarding a
 * per-thread stream offset), and draws locally, avoiding any shared mutable
 * state between threads.
 *
 * @note thrust::default_random_engine is currently a linear congruential engine.
 *       It is fast and cheap to copy but not cryptographically strong; that is an
 *       acceptable trade-off for Monte-Carlo particle sampling.
 * @see atlas::uniform_real_distribution, atlas::generate_standard_normal
 */
using default_random_engine = thrust::default_random_engine;

}