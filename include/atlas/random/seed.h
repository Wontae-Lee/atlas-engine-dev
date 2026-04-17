#pragma once

/**
 * @file seed.h
 * @brief Defines constants used for deterministic hash-based sampling and default random seeding.
 */

namespace atlas::seed {

/**
 * @brief Hash phase coefficient applied to the x-component of a 3D seed vector.
 *
 * This constant contributes to the phase term used in deterministic hash-like
 * scalar sampling functions.
 */
constexpr double RANDOM_HASH_PHASE_COEFF_X = 12.9898;

/**
 * @brief Hash phase coefficient applied to the y-component of a 3D seed vector.
 *
 * This constant contributes to the phase term used in deterministic hash-like
 * scalar sampling functions.
 */
constexpr double RANDOM_HASH_PHASE_COEFF_Y = 78.233;

/**
 * @brief Hash phase coefficient applied to the z-component of a 3D seed vector.
 *
 * This constant contributes to the phase term used in deterministic hash-like
 * scalar sampling functions.
 */
constexpr double RANDOM_HASH_PHASE_COEFF_Z = 37.719;

/**
 * @brief Scaling factor applied before folding a sine-based hash value into [0, 1).
 *
 * This value is used to amplify the intermediate sine result before extracting
 * its fractional part, helping produce decorrelated deterministic pseudo-random
 * samples.
 */
constexpr double RANDOM_HASH_VALUE_SCALE = 43758.5453;

/**
 * @brief Salt constant for the first diffuse sampling hash stream.
 *
 * This value is typically used to derive the first scalar sample for diffuse
 * hemisphere sampling while keeping it decorrelated from other hashed samples.
 */
constexpr double RANDOM_HASH_SALT_DIFFUSE_U1 = 0.31;

/**
 * @brief Salt constant for the second diffuse sampling hash stream.
 *
 * This value is typically used to derive the second scalar sample for diffuse
 * hemisphere sampling while keeping it decorrelated from other hashed samples.
 */
constexpr double RANDOM_HASH_SALT_DIFFUSE_U2 = 1.73;

/**
 * @brief Salt constant for stochastic diffuse/specular mixing decisions.
 *
 * This value is typically used to derive a deterministic hashed scalar that
 * selects between alternative interaction branches such as diffuse and specular
 * reflection.
 */
constexpr double RANDOM_HASH_SALT_MIX = 2.41;

/**
 * @brief Scaling factor applied to the normal vector contribution when building the mix hash seed.
 *
 * This constant helps perturb the hashed input used for branch-mixing decisions
 * so that the resulting sample is decorrelated from other seed constructions.
 */
constexpr double RANDOM_HASH_NORMAL_SCALE_FOR_MIX = 17.0;

/**
 * @brief Default unsigned integer seed value.
 *
 * This constant provides the default seed used when an explicit unsigned
 * integer seed is not supplied.
 */
constexpr unsigned int DEFAULT_UNSIGNED_INT_SEED = 0u;

} // namespace atlas::seed