#include "../utilities/test_utils.h"

#include <atlas/memory/copy.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/sampling/sampling.h>

#include <testkit/testkit.h>

#include <cmath>
#include <vector>

namespace {

using atlas::ExecutionPolicy;
using atlas::Vector3F;
using atlas::copy_device_to_host;
using atlas::default_random_engine;
using atlas::parallel_for;
using atlas::dot;
using atlas::normalize;
using atlas::build_orthonormal_basis;
using atlas::generate_standard_normal;
using atlas::sample_axis_count;
using atlas::sample_cosine_hemisphere;
using atlas::sample_directional_unit_vector;
using atlas::sample_hashed_unit_interval;
using atlas::sample_random_unit_vector;
using atlas::sample_uniform_hemisphere;
using atlas::tol;

} // namespace

TEST(Sampling, GenerateStandardNormalReturnsFiniteValue) {
    // Arrange: create a deterministic random engine.
    default_random_engine<float> engine(7u);

    // Act: generate one standard-normal sample.
    const auto value = generate_standard_normal<float>(engine);

    // Assert: generated values are finite.
    EXPECT_TRUE(std::isfinite(value));
}

TEST(Sampling, BuildOrthonormalBasisProducesOrthogonalVectors) {
    // Arrange: create a non-axis-aligned unit normal.
    const Vector3F normal = normalize(Vector3F(1, 2, 3));
#if defined(ATLAS_TASKING_CUDA)
    thrust::device_vector<Vector3F> basis(2);
    Vector3F* basis_ptr = thrust::raw_pointer_cast(basis.data());

    // Act: build the basis on the device.
    parallel_for<ExecutionPolicy::device>(
        0,
        1,
        [normal, basis_ptr] ATLAS_DEVICE(int) {
            Vector3F tangent;
            Vector3F bitangent;
            build_orthonormal_basis(normal, tangent, bitangent);
            basis_ptr[0] = tangent;
            basis_ptr[1] = bitangent;
        });

    std::vector<Vector3F> host_basis(2);
    copy_device_to_host(basis, host_basis.data(), host_basis.size());
    const Vector3F tangent = host_basis[0];
    const Vector3F bitangent = host_basis[1];
#else
    Vector3F tangent;
    Vector3F bitangent;

    // Act: build the basis on the host.
    build_orthonormal_basis(normal, tangent, bitangent);
#endif

    // Assert: the generated basis is orthonormal.
    EXPECT_NEAR(dot(normal, tangent), 0.0f, tol);
    EXPECT_NEAR(dot(normal, bitangent), 0.0f, tol);
    EXPECT_NEAR(dot(tangent, bitangent), 0.0f, tol);
    EXPECT_NEAR(tangent.length(), 1.0f, tol);
}

TEST(Sampling, UniformHemisphereSampleIsUnitLengthAndInHemisphere) {
    // Arrange: use the +Z hemisphere.
    const Vector3F normal(0, 0, 1);
#if defined(ATLAS_TASKING_CUDA)
    thrust::device_vector<Vector3F> device_sample(1);
    Vector3F* sample_ptr = thrust::raw_pointer_cast(device_sample.data());

    // Act: sample the hemisphere on the device.
    parallel_for<ExecutionPolicy::device>(
        0,
        1,
        [normal, sample_ptr] ATLAS_DEVICE(int) {
            sample_ptr[0] = sample_uniform_hemisphere(normal, 0.25f, 0.5f);
        });

    Vector3F sample;
    copy_device_to_host(device_sample, &sample, 1);
#else
    // Act: sample the hemisphere on the host.
    const auto sample = sample_uniform_hemisphere(normal, 0.25f, 0.5f);
#endif

    // Assert: the sample is normalized and lies in the requested hemisphere.
    EXPECT_NEAR(sample.length(), 1.0f, tol);
    EXPECT_GE(dot(sample, normal), 0.0f);
}

TEST(Sampling, CosineHemisphereSampleIsUnitLengthAndInHemisphere) {
    // Arrange: use the +Y hemisphere.
    const Vector3F normal(0, 1, 0);
#if defined(ATLAS_TASKING_CUDA)
    thrust::device_vector<Vector3F> device_sample(1);
    Vector3F* sample_ptr = thrust::raw_pointer_cast(device_sample.data());

    // Act: sample the hemisphere on the device.
    parallel_for<ExecutionPolicy::device>(
        0,
        1,
        [normal, sample_ptr] ATLAS_DEVICE(int) {
            sample_ptr[0] = sample_cosine_hemisphere(normal, 0.25f, 0.5f);
        });

    Vector3F sample;
    copy_device_to_host(device_sample, &sample, 1);
#else
    // Act: sample the hemisphere on the host.
    const auto sample = sample_cosine_hemisphere(normal, 0.25f, 0.5f);
#endif

    // Assert: the sample is normalized and lies in the requested hemisphere.
    EXPECT_NEAR(sample.length(), 1.0f, tol);
    EXPECT_GE(dot(sample, normal), 0.0f);
}

TEST(Sampling, RandomUnitVectorHasUnitLength) {
    // Arrange: create a deterministic random engine.
    default_random_engine<float> engine(9u);

    // Act: sample a random unit vector.
    const auto sample = sample_random_unit_vector<float>(engine);

    // Assert: the sample is normalized.
    EXPECT_NEAR(sample.length(), 1.0f, tol);
}

TEST(Sampling, DirectionalUnitVectorHasUnitLength) {
    // Arrange: create a deterministic engine and incoming direction.
    default_random_engine<float> engine(11u);
    const auto incoming = normalize(Vector3F(1, 1, 1));

    // Act: sample a directional unit vector.
    const auto sample = sample_directional_unit_vector(incoming, 2.0f, engine);

    // Assert: the sample is normalized.
    EXPECT_NEAR(sample.length(), 1.0f, tol);
}

TEST(Sampling, SampleAxisCountHandlesValidAndInvalidInputs) {
    // Assert: valid ranges include both endpoints, while invalid inputs produce no samples.
    EXPECT_EQ(sample_axis_count(0.0f, 1.0f, 0.5f), 3);
    EXPECT_EQ(sample_axis_count(0.0f, 1.0f, 0.0f), 0);
    EXPECT_EQ(sample_axis_count(2.0f, 1.0f, 0.5f), 0);
}

TEST(Sampling, HashedUnitIntervalIsDeterministicAndBounded) {
    // Arrange: create a stable hash seed vector.
    const Vector3F seed(1.0f, 2.0f, 3.0f);

    // Act: sample with identical and different salt values.
    const auto value0 = sample_hashed_unit_interval(seed, 0.5f);
    const auto value1 = sample_hashed_unit_interval(seed, 0.5f);
    const auto value2 = sample_hashed_unit_interval(seed, 0.6f);

    // Assert: hash sampling is deterministic and bounded to [0, 1).
    EXPECT_EQ(value0, value1);
    EXPECT_GE(value0, 0.0f);
    EXPECT_LT(value0, 1.0f);
    EXPECT_NE(value0, value2);
}
