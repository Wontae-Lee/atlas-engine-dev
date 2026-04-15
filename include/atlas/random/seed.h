#pragma once
#include <atlas/core/macros.h>

namespace atlas::seed {

constexpr double random_hash_phase_coeff_x = 12.9898;

constexpr double random_hash_phase_coeff_y = 78.233;

constexpr double random_hash_phase_coeff_z = 37.719;

constexpr double random_hash_value_scale = 43758.5453;

constexpr double random_hash_salt_diffuse_u1 = 0.31;

constexpr double random_hash_salt_diffuse_u2 = 1.73;

constexpr double random_hash_salt_mix = 2.41;

constexpr double random_hash_normal_scale_for_mix = 17.0;

constexpr unsigned int default_unsigned_int_seed = 0.0;

}