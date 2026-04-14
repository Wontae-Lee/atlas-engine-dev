#pragma once
#include <atlas/core/macros.h>

/**
 * @file seed.h
 * @brief Common random-related constants used by atlas random and sampling utilities.
 *
 * @details
 * This header centralizes numeric constants that are shared by deterministic
 * pseudo-random helper routines and sampling-related utilities.
 *
 * The constants below are intended for hash-style mappings that convert
 * geometric seeds into reproducible scalar values in the unit interval.
 *
 * @warning
 * These values are implementation constants rather than physically meaningful
 * parameters. Changing them will change deterministic sampling outcomes.
 */

namespace atlas::seed {

/**
 * @brief Scrambling coefficient applied to the x-component of a 3D hash seed.
 */
constexpr double random_hash_phase_coeff_x = 12.9898;

/**
 * @brief Scrambling coefficient applied to the y-component of a 3D hash seed.
 */
constexpr double random_hash_phase_coeff_y = 78.233;

/**
 * @brief Scrambling coefficient applied to the z-component of a 3D hash seed.
 */
constexpr double random_hash_phase_coeff_z = 37.719;

/**
 * @brief Post-sine scaling factor used before extracting the fractional part.
 */
constexpr double random_hash_value_scale = 43758.5453;

/**
 * @brief Salt used to derive the first hashed scalar for diffuse sampling.
 */
constexpr double random_hash_salt_diffuse_u1 = 0.31;

/**
 * @brief Salt used to derive the second hashed scalar for diffuse sampling.
 */
constexpr double random_hash_salt_diffuse_u2 = 1.73;

/**
 * @brief Salt used to derive the hashed scalar for specular/diffuse mixing.
 */
constexpr double random_hash_salt_mix = 2.41;

/**
 * @brief Scale applied to the surface normal when constructing the mix seed.
 */
constexpr double random_hash_normal_scale_for_mix = 17.0;

/**
 * @brief Default seed value.
 */
constexpr unsigned int default_unsigned_int_seed = 0.0;

} // namespace atlas::seed
