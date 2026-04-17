#include "../utilities/tests_utils.h"

#include <atlas/random/seed.h>

#include <gtest/gtest.h>

TEST(Seed, HashConstantsRemainStable) {
    EXPECT_DOUBLE_EQ(atlas::seed::RANDOM_HASH_PHASE_COEFF_X, 12.9898);
    EXPECT_DOUBLE_EQ(atlas::seed::RANDOM_HASH_PHASE_COEFF_Y, 78.233);
    EXPECT_DOUBLE_EQ(atlas::seed::RANDOM_HASH_PHASE_COEFF_Z, 37.719);
    EXPECT_DOUBLE_EQ(atlas::seed::RANDOM_HASH_VALUE_SCALE, 43758.5453);
    EXPECT_DOUBLE_EQ(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1, 0.31);
    EXPECT_DOUBLE_EQ(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2, 1.73);
    EXPECT_DOUBLE_EQ(atlas::seed::RANDOM_HASH_SALT_MIX, 2.41);
    EXPECT_DOUBLE_EQ(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX, 17.0);
    EXPECT_EQ(atlas::seed::DEFAULT_UNSIGNED_INT_SEED, 0u);
}
