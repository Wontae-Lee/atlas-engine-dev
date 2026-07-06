#include <atlas/sampling/sampling.h>

#include <atlas/random/default_random_engine.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::Vector3;
using atlas::build_orthonormal_basis;
using atlas::default_random_engine;
using atlas::dot;
using atlas::generate_standard_normal;
using atlas::sample_axis_count;
using atlas::sample_cosine_hemisphere;
using atlas::sample_directional_unit_vector;
using atlas::sample_hashed_index;
using atlas::sample_hashed_unit_interval;
using atlas::sample_random_unit_vector;
using atlas::sample_uniform_hemisphere;
using atlas::tol;

}

TEST(Sampling, GenerateStandardNormalReturnsFiniteValue) {
    default_random_engine engine(7u);

    const float value = generate_standard_normal(engine);

    EXPECT_TRUE(std::isfinite(value));
}

TEST(Sampling, BuildOrthonormalBasisProducesOrthogonalVectors) {
    const Vector3 normal = Vector3(1.0f, 2.0f, 3.0f).normalized();

    Vector3 tangent;
    Vector3 bitangent;
    build_orthonormal_basis(normal, tangent, bitangent);

    EXPECT_NEAR(dot(normal, tangent), 0.0f, tol);
    EXPECT_NEAR(dot(normal, bitangent), 0.0f, tol);
    EXPECT_NEAR(dot(tangent, bitangent), 0.0f, tol);
    EXPECT_NEAR(tangent.length(), 1.0f, tol);
}

TEST(Sampling, UniformHemisphereSampleIsUnitLengthAndInHemisphere) {
    const Vector3 normal(0.0f, 0.0f, 1.0f);

    const Vector3 sample = sample_uniform_hemisphere(normal, 0.25f, 0.5f);

    EXPECT_NEAR(sample.length(), 1.0f, tol);
    EXPECT_GE(dot(sample, normal), 0.0f);
}

TEST(Sampling, CosineHemisphereSampleIsUnitLengthAndInHemisphere) {
    const Vector3 normal(0.0f, 1.0f, 0.0f);

    const Vector3 sample = sample_cosine_hemisphere(normal, 0.25f, 0.5f);

    EXPECT_NEAR(sample.length(), 1.0f, tol);
    EXPECT_GE(dot(sample, normal), 0.0f);
}

TEST(Sampling, RandomUnitVectorHasUnitLength) {
    default_random_engine engine(9u);

    const Vector3 sample = sample_random_unit_vector(engine);

    EXPECT_NEAR(sample.length(), 1.0f, tol);
}

TEST(Sampling, DirectionalUnitVectorHasUnitLength) {
    default_random_engine engine(11u);
    const Vector3 incoming = Vector3(1.0f, 1.0f, 1.0f).normalized();

    const Vector3 sample = sample_directional_unit_vector(incoming, 2.0f, engine);

    EXPECT_NEAR(sample.length(), 1.0f, tol);
}

TEST(Sampling, SampleAxisCountHandlesValidAndInvalidInputs) {
    EXPECT_EQ(sample_axis_count(0.0f, 1.0f, 0.5f), 3);
    EXPECT_EQ(sample_axis_count(0.0f, 1.0f, 0.0f), 0);
    EXPECT_EQ(sample_axis_count(2.0f, 1.0f, 0.5f), 0);
}

TEST(Sampling, HashedUnitIntervalIsDeterministicAndBounded) {
    const Vector3 seed(1.0f, 2.0f, 3.0f);

    const float value0 = sample_hashed_unit_interval(seed, 0.5f);
    const float value1 = sample_hashed_unit_interval(seed, 0.5f);
    const float value2 = sample_hashed_unit_interval(seed, 0.6f);

    EXPECT_EQ(value0, value1);
    EXPECT_GE(value0, 0.0f);
    EXPECT_LT(value0, 1.0f);
    EXPECT_NE(value0, value2);
}

TEST(Sampling, HashedIndexStaysInsideUpperBound) {
    for (int i = 0; i < 32; ++i) {
        const int index = sample_hashed_index(i, 5, 123u);
        EXPECT_GE(index, 0);
        EXPECT_LT(index, 5);
    }

    EXPECT_EQ(sample_hashed_index(3, 0, 123u), 0);
}
