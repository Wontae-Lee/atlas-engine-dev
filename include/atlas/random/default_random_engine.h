#pragma once

/**
 * @file default_random_engine.h
 * @brief Declares the backend-portable default pseudo-random engine alias used throughout Atlas.
 *
 * @details
 * This header defines @ref atlas::default_random_engine, a compile-time alias
 * that selects the default random-engine type used by Atlas depending on the
 * active tasking backend.
 *
 * ## Purpose
 * Atlas code often needs a lightweight random-engine type that can be used in:
 * - host-side sampling code,
 * - backend-portable generator operators,
 * - CUDA device-capable paths when available.
 *
 * Rather than hard-coding a specific standard-library or Thrust engine in each
 * component, this header centralizes the backend selection behind a single alias.
 *
 * ## Backend selection
 * The chosen engine depends on whether `ATLAS_TASKING_CUDA` is defined:
 * - when CUDA tasking is enabled, the alias resolves to `thrust::default_random_engine`,
 * - otherwise, it resolves to `std::default_random_engine`.
 *
 * ## Design intent
 * This abstraction allows higher-level Atlas components to:
 * - remain backend-agnostic,
 * - use a consistent engine name across host-only and CUDA-enabled builds,
 * - avoid scattering conditional compilation logic throughout generator code.
 *
 * ## Notes
 * - The statistical properties, state layout, and exact reproducibility behavior
 *   depend on the selected backend engine implementation.
 * - Users of this alias should treat it as a convenient default engine, not as a
 *   guarantee of cross-backend bitwise-identical random sequences.
 *
 * ---
 */

#ifdef ATLAS_TASKING_CUDA
#include <thrust/random.h>
#else
#include <random>
#endif

namespace atlas {

/**
 * @brief Backend-portable alias for the default pseudo-random engine.
 *
 * @details
 * This alias resolves to:
 * - `thrust::default_random_engine` in CUDA-enabled builds,
 * - `std::default_random_engine` in non-CUDA builds.
 *
 * It is intended to provide a uniform random-engine name for Atlas components
 * that need lightweight pseudo-random number generation across multiple
 * backends.
 *
 * @tparam T Floating-point scalar type associated with the surrounding usage context.
 *
 * @note
 * The template parameter @p T is not used directly by the alias itself, but is
 * preserved to keep the naming and usage style consistent with other
 * Atlas template-based utility types.
 */
template <typename T>
#ifdef ATLAS_TASKING_CUDA
using default_random_engine = thrust::default_random_engine;
#else
using default_random_engine = std::default_random_engine;
#endif

} // namespace atlas