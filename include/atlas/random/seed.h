#pragma once

#include <cstdint>

namespace atlas {

constexpr float RANDOM_HASH_PHASE_COEFF_X = 12.9898f;

constexpr float RANDOM_HASH_PHASE_COEFF_Y = 78.233f;

constexpr float RANDOM_HASH_PHASE_COEFF_Z = 37.719f;

constexpr float RANDOM_HASH_VALUE_SCALE = 43758.5453f;

constexpr float RANDOM_HASH_SALT_DIFFUSE_U1 = 0.31f;

constexpr float RANDOM_HASH_SALT_DIFFUSE_U2 = 1.73f;

constexpr float RANDOM_HASH_SALT_MIX = 2.41f;

constexpr float RANDOM_HASH_NORMAL_SCALE_FOR_MIX = 17.0f;

constexpr unsigned int DEFAULT_UNSIGNED_INT_SEED = 0u;

constexpr int RANDOM_HASH_UNIT_INTERVAL_SHIFT = 11;

constexpr float RANDOM_HASH_UNIT_INTERVAL_SCALE = 1.0f / 9007199254740992.0f;

constexpr std::uint64_t DSMC_CELL_STREAM_MULTIPLIER = 0x9e3779b97f4a7c15ull;

constexpr std::uint64_t DSMC_COLLISION_LHS_SALT = 0x632be59bd9b4e019ull;

constexpr std::uint64_t DSMC_COLLISION_RHS_SALT = 0x85157af5ull;

constexpr std::uint64_t DSMC_COLLISION_ACCEPT_SALT = 0xda942042e4dd58b5ull;

constexpr int SHUFFLE_HASH_FIRST_SHIFT = 30;

constexpr int SHUFFLE_HASH_SECOND_SHIFT = 27;

constexpr int SHUFFLE_HASH_FINAL_SHIFT = 31;

constexpr std::uint64_t SHUFFLE_HASH_INDEX_OFFSET = 0x9e3779b97f4a7c15ull;

constexpr std::uint64_t SHUFFLE_HASH_FIRST_MULTIPLIER = 0xbf58476d1ce4e5b9ull;

constexpr std::uint64_t SHUFFLE_HASH_SECOND_MULTIPLIER = 0x94d049bb133111ebull;

constexpr unsigned MORTON_EXPAND_BITS_FIRST_MULTIPLIER = 0x00010001u;

constexpr unsigned MORTON_EXPAND_BITS_FIRST_MASK = 0xFF0000FFu;

constexpr unsigned MORTON_EXPAND_BITS_SECOND_MULTIPLIER = 0x00000101u;

constexpr unsigned MORTON_EXPAND_BITS_SECOND_MASK = 0x0F00F00Fu;

constexpr unsigned MORTON_EXPAND_BITS_THIRD_MULTIPLIER = 0x00000011u;

constexpr unsigned MORTON_EXPAND_BITS_THIRD_MASK = 0xC30C30C3u;

constexpr unsigned MORTON_EXPAND_BITS_FINAL_MULTIPLIER = 0x00000005u;

constexpr unsigned MORTON_EXPAND_BITS_FINAL_MASK = 0x49249249u;

}
