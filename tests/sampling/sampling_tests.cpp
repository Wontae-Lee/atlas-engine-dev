#include "../utilities/tests_utils.h"

#include <atlas/random/default_random_engine.h>
#include <atlas/sampling/sampling.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-4);

} // namespace

TEST(Sampling, GenerateStandardNormalReturnsFiniteValue) {
    atlas::default_random_engine<T> engine(7u);

    const auto value = atlas::sampling::generate_standard_normal<T>(engine);

    EXPECT_TRUE(std::isfinite(value));
}

TEST(Sampling, BuildOrthonormalBasisProducesOrthogonalVectors) {
    const Vec3 normal = atlas::math::normalize(Vec3(1, 2, 3));
    Vec3 tangent;
    Vec3 bitangent;

    atlas::sampling::build_orthonormal_basis(normal, tangent, bitangent);

    EXPECT_NEAR(atlas::math::dot(normal, tangent), 0.0f, 1e-3f);
    EXPECT_NEAR(atlas::math::dot(normal, bitangent), 0.0f, 1e-3f);
    EXPECT_NEAR(atlas::math::dot(tangent, bitangent), 0.0f, 1e-3f);
    EXPECT_NEAR(tangent.length(), 1.0f, 1e-3f);
}

TEST(Sampling, UniformHemisphereSampleIsUnitLengthAndInHemisphere) {
    const Vec3 normal(0, 0, 1);
    const auto sample = atlas::sampling::sample_uniform_hemisphere(normal, 0.25f, 0.5f);

    EXPECT_NEAR(sample.length(), 1.0f, 1e-3f);
    EXPECT_GE(atlas::math::dot(sample, normal), 0.0f);
}

TEST(Sampling, CosineHemisphereSampleIsUnitLengthAndInHemisphere) {
    const Vec3 normal(0, 1, 0);
    const auto sample = atlas::sampling::sample_cosine_hemisphere(normal, 0.25f, 0.5f);

    EXPECT_NEAR(sample.length(), 1.0f, 1e-3f);
    EXPECT_GE(atlas::math::dot(sample, normal), 0.0f);
}

TEST(Sampling, RandomUnitVectorHasUnitLength) {
    atlas::default_random_engine<T> engine(9u);

    const auto sample = atlas::sampling::sample_random_unit_vector<T>(engine);

    EXPECT_NEAR(sample.length(), 1.0f, 1e-3f);
}

TEST(Sampling, DirectionalUnitVectorHasUnitLength) {
    atlas::default_random_engine<T> engine(11u);
    const auto incoming = atlas::math::normalize(Vec3(1, 1, 1));

    const auto sample = atlas::sampling::sample_directional_unit_vector(incoming, 2.0f, engine);

    EXPECT_NEAR(sample.length(), 1.0f, 1e-3f);
}

TEST(Sampling, SampleAxisCountHandlesValidAndInvalidInputs) {
    EXPECT_EQ(atlas::sampling::sample_axis_count(0.0f, 1.0f, 0.5f), 3);
    EXPECT_EQ(atlas::sampling::sample_axis_count(0.0f, 1.0f, 0.0f), 0);
    EXPECT_EQ(atlas::sampling::sample_axis_count(2.0f, 1.0f, 0.5f), 0);
}

TEST(Sampling, HashedUnitIntervalIsDeterministicAndBounded) {
    const Vec3 seed(1.0f, 2.0f, 3.0f);

    const auto value0 = atlas::sampling::sample_hashed_unit_interval(seed, 0.5f);
    const auto value1 = atlas::sampling::sample_hashed_unit_interval(seed, 0.5f);
    const auto value2 = atlas::sampling::sample_hashed_unit_interval(seed, 0.6f);

    EXPECT_EQ(value0, value1);
    EXPECT_GE(value0, 0.0f);
    EXPECT_LT(value0, 1.0f);
    EXPECT_NE(value0, value2);
}
