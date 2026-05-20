#pragma once

#include <cstdint>

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

constexpr int RANDOM_HASH_UNIT_INTERVAL_SHIFT = 11;

constexpr double RANDOM_HASH_UNIT_INTERVAL_SCALE = 1.0 / 9007199254740992.0;

constexpr std::uint64_t DSMC_CELL_STREAM_MULTIPLIER = 0x9e3779b97f4a7c15ull;

constexpr std::uint64_t DSMC_COLLISION_LHS_SALT = 0x632be59bd9b4e019ull;

constexpr std::uint64_t DSMC_COLLISION_RHS_SALT = 0x85157af5ull;

constexpr std::uint64_t DSMC_COLLISION_ACCEPT_SALT = 0xda942042e4dd58b5ull;

/**
 * @brief Bit shift applied during the first shuffle hash mixing stage.
 */
constexpr int SHUFFLE_HASH_FIRST_SHIFT = 30;

/**
 * @brief Bit shift applied during the second shuffle hash mixing stage.
 */
constexpr int SHUFFLE_HASH_SECOND_SHIFT = 27;

/**
 * @brief Bit shift applied during the final shuffle hash mixing stage.
 */
constexpr int SHUFFLE_HASH_FINAL_SHIFT = 31;

/**
 * @brief Additive offset applied to shuffle indices before hash mixing.
 */
constexpr std::uint64_t SHUFFLE_HASH_INDEX_OFFSET = 0x9e3779b97f4a7c15ull;

/**
 * @brief First multiplicative shuffle hash mixing constant.
 */
constexpr std::uint64_t SHUFFLE_HASH_FIRST_MULTIPLIER = 0xbf58476d1ce4e5b9ull;

/**
 * @brief Second multiplicative shuffle hash mixing constant.
 */
constexpr std::uint64_t SHUFFLE_HASH_SECOND_MULTIPLIER = 0x94d049bb133111ebull;

/**
 * @brief First multiplicative constant used to expand Morton-code coordinate bits.
 */
constexpr unsigned MORTON_EXPAND_BITS_FIRST_MULTIPLIER = 0x00010001u;

/**
 * @brief First mask used to expand Morton-code coordinate bits.
 */
constexpr unsigned MORTON_EXPAND_BITS_FIRST_MASK = 0xFF0000FFu;

/**
 * @brief Second multiplicative constant used to expand Morton-code coordinate bits.
 */
constexpr unsigned MORTON_EXPAND_BITS_SECOND_MULTIPLIER = 0x00000101u;

/**
 * @brief Second mask used to expand Morton-code coordinate bits.
 */
constexpr unsigned MORTON_EXPAND_BITS_SECOND_MASK = 0x0F00F00Fu;

/**
 * @brief Third multiplicative constant used to expand Morton-code coordinate bits.
 */
constexpr unsigned MORTON_EXPAND_BITS_THIRD_MULTIPLIER = 0x00000011u;

/**
 * @brief Third mask used to expand Morton-code coordinate bits.
 */
constexpr unsigned MORTON_EXPAND_BITS_THIRD_MASK = 0xC30C30C3u;

/**
 * @brief Final multiplicative constant used to expand Morton-code coordinate bits.
 */
constexpr unsigned MORTON_EXPAND_BITS_FINAL_MULTIPLIER = 0x00000005u;

/**
 * @brief Final mask used to expand Morton-code coordinate bits.
 */
constexpr unsigned MORTON_EXPAND_BITS_FINAL_MASK = 0x49249249u;

} // namespace atlas::seed
