#pragma once

#include <cstdint>

namespace atlas {

/**
 * @file seed.h
 * @brief Named magic numbers for the engine's stateless hashing and mixing.
 *
 * Atlas prefers *stateless* pseudo-randomness on the device: rather than carrying
 * an RNG object across kernels, most partner-selection and spatial code derives
 * a value directly from integer indices, cell ids, and time-step "streams" through
 * a hash. Centralizing every constant here keeps those hashes reproducible across
 * a run (and across restarts) and documents the otherwise opaque bit patterns.
 *
 * Where a hash of physical state would bias the result — notably the DSMC scatter
 * angle, whose seed would otherwise be the collision's own velocities — the code
 * instead seeds an @ref atlas::default_random_engine from one of the stream salts
 * below and draws from it.
 *
 * The values fall into four families:
 *   - the sine-hash "phase"/"scale" constants (GLSL-style `fract(sin(dot)·k)`),
 *   - the SplitMix64 finalizer constants used by @ref atlas::shuffle_key,
 *   - per-purpose salts that decorrelate distinct draws sharing one stream, and
 *   - the bit-spreading multipliers/masks that build 30-bit Morton codes.
 */

/**
 * @brief X-axis dot-product weight of the sine hash's phase.
 *
 * The classic GLSL one-liner `fract(sin(dot(seed, k)) * scale)` mixes a 3D seed
 * into a pseudo-random unit-interval value; these coefficients are the `k` vector.
 * @see atlas::sample_hashed_unit_interval(const Float3&, float)
 */
constexpr float RANDOM_HASH_PHASE_COEFF_X = 12.9898f;

/**
 * @brief Y-axis dot-product weight of the sine hash's phase.
 * @see RANDOM_HASH_PHASE_COEFF_X
 */
constexpr float RANDOM_HASH_PHASE_COEFF_Y = 78.233f;

/**
 * @brief Z-axis dot-product weight of the sine hash's phase.
 * @see RANDOM_HASH_PHASE_COEFF_X
 */
constexpr float RANDOM_HASH_PHASE_COEFF_Z = 37.719f;

/**
 * @brief Large multiplier applied to `sin(phase)` before taking the fractional part.
 *
 * Inflating the sine so its integer part dwarfs its fraction is what makes the
 * final `value - floor(value)` behave like a uniform draw.
 */
constexpr float RANDOM_HASH_VALUE_SCALE = 43758.5453f;

/**
 * @brief Salt added to the sine-hash phase when drawing the first diffuse variate.
 *
 * The isothermal collider needs two independent unit-interval values from one
 * geometric seed; adding distinct salts decorrelates them.
 */
constexpr float RANDOM_HASH_SALT_DIFFUSE_U1 = 0.31f;

/**
 * @brief Salt added to the sine-hash phase when drawing the second diffuse variate.
 * @see RANDOM_HASH_SALT_DIFFUSE_U1
 */
constexpr float RANDOM_HASH_SALT_DIFFUSE_U2 = 1.73f;

/**
 * @brief Salt added to the sine-hash phase when choosing the specular/diffuse mix.
 */
constexpr float RANDOM_HASH_SALT_MIX = 2.41f;

/**
 * @brief Weight that folds the surface normal into a hash seed.
 *
 * Scaling the normal before adding it to a position seed ensures two points that
 * share a location but differ in orientation hash to different values.
 */
constexpr float RANDOM_HASH_NORMAL_SCALE_FOR_MIX = 17.0f;

/**
 * @brief Default RNG seed for the particle generators.
 *
 * Generators expose a `with_seed` setter but fall back to this deterministic value
 * so that an unconfigured run is still reproducible.
 */
constexpr unsigned int DEFAULT_UNSIGNED_INT_SEED = 0u;

/**
 * @brief Right-shift discarding the low mantissa bits of a 64-bit hash.
 *
 * The top `64 - 11 = 53` bits of a shuffle key are kept and scaled by
 * @ref RANDOM_HASH_UNIT_INTERVAL_SCALE to form a double-precision-exact draw in
 * `[0, 1)`. Eleven is `64 - 53`, i.e. the number of bits above a double's mantissa.
 * @see atlas::sample_hashed_unit_interval(int, std::uint64_t)
 */
constexpr int RANDOM_HASH_UNIT_INTERVAL_SHIFT = 11;

/**
 * @brief Reciprocal of `2^53`, mapping a 53-bit integer onto `[0, 1)`.
 * @see RANDOM_HASH_UNIT_INTERVAL_SHIFT
 */
constexpr float RANDOM_HASH_UNIT_INTERVAL_SCALE = 1.0f / 9007199254740992.0f;

/**
 * @brief Per-cell stream stride for the DSMC solver (the golden-ratio odd constant).
 *
 * Multiplying a cell id by this odd constant and adding it to the time-step seed
 * gives each cell a well-separated random stream, so neighboring cells do not draw
 * correlated collision partners.
 */
constexpr std::uint64_t DSMC_CELL_STREAM_MULTIPLIER = 0x9e3779b97f4a7c15ull;

/**
 * @brief Salt distinguishing the left-hand collision-partner draw within a cell.
 */
constexpr std::uint64_t DSMC_COLLISION_LHS_SALT = 0x632be59bd9b4e019ull;

/**
 * @brief Salt distinguishing the right-hand collision-partner draw within a cell.
 */
constexpr std::uint64_t DSMC_COLLISION_RHS_SALT = 0x85157af5ull;

/**
 * @brief Salt for the NTC acceptance test of a candidate collision pair.
 */
constexpr std::uint64_t DSMC_COLLISION_ACCEPT_SALT = 0xda942042e4dd58b5ull;

/**
 * @brief Salt seeding the per-collision scatter engine.
 *
 * @ref atlas::dsmc_scatter draws its deflection and azimuth from a
 * @ref atlas::default_random_engine rather than from a hash of the pair's velocities.
 * The solver seeds that engine by folding this salt into the collision's `(cell, stream)`
 * pair through @ref atlas::shuffle_key, which keeps the scatter stream separate from the
 * partner-selection and acceptance draws that share the same stream base.
 */
constexpr std::uint64_t DSMC_COLLISION_SCATTER_SALT = 0xc2b2ae3d27d4eb4full;

/**
 * @brief First xor-shift amount of the SplitMix64 finalizer in @ref atlas::shuffle_key.
 */
constexpr int SHUFFLE_HASH_FIRST_SHIFT = 30;

/**
 * @brief Second xor-shift amount of the SplitMix64 finalizer in @ref atlas::shuffle_key.
 */
constexpr int SHUFFLE_HASH_SECOND_SHIFT = 27;

/**
 * @brief Final xor-shift amount of the SplitMix64 finalizer in @ref atlas::shuffle_key.
 */
constexpr int SHUFFLE_HASH_FINAL_SHIFT = 31;

/**
 * @brief Increment folded into the index before hashing (golden-ratio odd constant).
 *
 * Bit-identical to @ref DSMC_CELL_STREAM_MULTIPLIER; kept separate because it plays
 * the SplitMix64 "increment" role here rather than a stream stride.
 */
constexpr std::uint64_t SHUFFLE_HASH_INDEX_OFFSET = 0x9e3779b97f4a7c15ull;

/**
 * @brief First avalanche multiplier of the SplitMix64 finalizer.
 */
constexpr std::uint64_t SHUFFLE_HASH_FIRST_MULTIPLIER = 0xbf58476d1ce4e5b9ull;

/**
 * @brief Second avalanche multiplier of the SplitMix64 finalizer.
 */
constexpr std::uint64_t SHUFFLE_HASH_SECOND_MULTIPLIER = 0x94d049bb133111ebull;

/**
 * @brief First multiplier of the Morton bit-spread (spaces bits by three).
 *
 * The four multiplier/mask pairs implement the standard "expand bits" step that
 * interleaves a 10-bit coordinate into a 30-bit Morton code for the LBVH.
 * @see src/atlas/spatial/bounding_volume_hierarchy/lbvh.cu
 */
constexpr unsigned MORTON_EXPAND_BITS_FIRST_MULTIPLIER = 0x00010001u;

/**
 * @brief First mask of the Morton bit-spread.
 * @see MORTON_EXPAND_BITS_FIRST_MULTIPLIER
 */
constexpr unsigned MORTON_EXPAND_BITS_FIRST_MASK = 0xFF0000FFu;

/**
 * @brief Second multiplier of the Morton bit-spread.
 * @see MORTON_EXPAND_BITS_FIRST_MULTIPLIER
 */
constexpr unsigned MORTON_EXPAND_BITS_SECOND_MULTIPLIER = 0x00000101u;

/**
 * @brief Second mask of the Morton bit-spread.
 * @see MORTON_EXPAND_BITS_FIRST_MULTIPLIER
 */
constexpr unsigned MORTON_EXPAND_BITS_SECOND_MASK = 0x0F00F00Fu;

/**
 * @brief Third multiplier of the Morton bit-spread.
 * @see MORTON_EXPAND_BITS_FIRST_MULTIPLIER
 */
constexpr unsigned MORTON_EXPAND_BITS_THIRD_MULTIPLIER = 0x00000011u;

/**
 * @brief Third mask of the Morton bit-spread.
 * @see MORTON_EXPAND_BITS_FIRST_MULTIPLIER
 */
constexpr unsigned MORTON_EXPAND_BITS_THIRD_MASK = 0xC30C30C3u;

/**
 * @brief Final multiplier of the Morton bit-spread.
 * @see MORTON_EXPAND_BITS_FIRST_MULTIPLIER
 */
constexpr unsigned MORTON_EXPAND_BITS_FINAL_MULTIPLIER = 0x00000005u;

/**
 * @brief Final mask of the Morton bit-spread (isolates the interleaved bits).
 * @see MORTON_EXPAND_BITS_FIRST_MULTIPLIER
 */
constexpr unsigned MORTON_EXPAND_BITS_FINAL_MASK = 0x49249249u;

}
