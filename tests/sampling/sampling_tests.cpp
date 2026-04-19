#include "../utilities/tests_utils.h"

#include <atlas/memory/copy.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/sampling/sampling.h>

#include <testkit/testkit.h>

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
#if defined(ATLAS_TASKING_CUDA)
    thrust::device_vector<Vec3> basis(2);
    Vec3* basis_ptr = thrust::raw_pointer_cast(basis.data());

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        1,
        [normal, basis_ptr] ATLAS_DEVICE(int) {
            Vec3 tangent;
            Vec3 bitangent;
            atlas::sampling::build_orthonormal_basis(normal, tangent, bitangent);
            basis_ptr[0] = tangent;
            basis_ptr[1] = bitangent;
        });

    std::vector<Vec3> host_basis(2);
    atlas::copy_device_to_host(basis, host_basis.data(), host_basis.size());
    const Vec3 tangent = host_basis[0];
    const Vec3 bitangent = host_basis[1];
#else
    Vec3 tangent;
    Vec3 bitangent;

    atlas::sampling::build_orthonormal_basis(normal, tangent, bitangent);
#endif

    EXPECT_NEAR(atlas::math::dot(normal, tangent), 0.0f, 1e-3f);
    EXPECT_NEAR(atlas::math::dot(normal, bitangent), 0.0f, 1e-3f);
    EXPECT_NEAR(atlas::math::dot(tangent, bitangent), 0.0f, 1e-3f);
    EXPECT_NEAR(tangent.length(), 1.0f, 1e-3f);
}

TEST(Sampling, UniformHemisphereSampleIsUnitLengthAndInHemisphere) {
    const Vec3 normal(0, 0, 1);
#if defined(ATLAS_TASKING_CUDA)
    thrust::device_vector<Vec3> device_sample(1);
    Vec3* sample_ptr = thrust::raw_pointer_cast(device_sample.data());

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        1,
        [normal, sample_ptr] ATLAS_DEVICE(int) {
            sample_ptr[0] = atlas::sampling::sample_uniform_hemisphere(normal, 0.25f, 0.5f);
        });

    Vec3 sample;
    atlas::copy_device_to_host(device_sample, &sample, 1);
#else
    const auto sample = atlas::sampling::sample_uniform_hemisphere(normal, 0.25f, 0.5f);
#endif

    EXPECT_NEAR(sample.length(), 1.0f, 1e-3f);
    EXPECT_GE(atlas::math::dot(sample, normal), 0.0f);
}

TEST(Sampling, CosineHemisphereSampleIsUnitLengthAndInHemisphere) {
    const Vec3 normal(0, 1, 0);
#if defined(ATLAS_TASKING_CUDA)
    thrust::device_vector<Vec3> device_sample(1);
    Vec3* sample_ptr = thrust::raw_pointer_cast(device_sample.data());

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        1,
        [normal, sample_ptr] ATLAS_DEVICE(int) {
            sample_ptr[0] = atlas::sampling::sample_cosine_hemisphere(normal, 0.25f, 0.5f);
        });

    Vec3 sample;
    atlas::copy_device_to_host(device_sample, &sample, 1);
#else
    const auto sample = atlas::sampling::sample_cosine_hemisphere(normal, 0.25f, 0.5f);
#endif

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
