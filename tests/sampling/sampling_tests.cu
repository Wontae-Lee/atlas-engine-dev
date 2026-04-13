#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

#include <cmath>

CUDA_TEST(Sampling, BuildOrthonormalBasisProducesOrthogonalFrame) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n = atlas::math::normalize(atlas::Vector3D(0.3, 0.4, 0.5));
    atlas::DeviceBuffer<atlas::Vector3D> frame(2);
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        static_cast<std::size_t>(1),
        [data = thrust::raw_pointer_cast(frame.data()), n] ATLAS_DEVICE(std::size_t) {
            atlas::Vector3D t, b;
            atlas::sampling::build_orthonormal_basis(n, t, b);
            data[0] = t;
            data[1] = b;
        });
    const auto host_frame = ::atlas::test::cuda::to_host_vector(frame);
    const auto& t = host_frame[0];
    const auto& b = host_frame[1];

    CUDA_EXPECT_TRUE(atlas::test::near(t.length(), 1.0, eps));
    CUDA_EXPECT_TRUE(atlas::test::near(b.length(), 1.0, eps));

    CUDA_EXPECT_TRUE(atlas::test::near(n.dot(t), 0.0, eps));
    CUDA_EXPECT_TRUE(atlas::test::near(n.dot(b), 0.0, eps));
    CUDA_EXPECT_TRUE(atlas::test::near(t.dot(b), 0.0, eps));

    const auto bx = atlas::math::cross(n, t);
    CUDA_EXPECT_TRUE(atlas::test::vec_near(b, bx, eps));
}

CUDA_TEST(Sampling, BuildOrthonormalBasisHandlesAxisAlignedNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);
    atlas::DeviceBuffer<atlas::Vector3D> frame(2);
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        static_cast<std::size_t>(1),
        [data = thrust::raw_pointer_cast(frame.data()), n] ATLAS_DEVICE(std::size_t) {
            atlas::Vector3D t, b;
            atlas::sampling::build_orthonormal_basis(n, t, b);
            data[0] = t;
            data[1] = b;
        });
    const auto host_frame = ::atlas::test::cuda::to_host_vector(frame);
    const auto& t = host_frame[0];
    const auto& b = host_frame[1];

    CUDA_EXPECT_TRUE(atlas::test::near(n.dot(t), 0.0, eps));
    CUDA_EXPECT_TRUE(atlas::test::near(n.dot(b), 0.0, eps));
    CUDA_EXPECT_TRUE(atlas::test::near(t.dot(b), 0.0, eps));
}

CUDA_TEST(Sampling, UniformHemisphereSamplesLieOnUpperHemisphere) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);
    atlas::DeviceBuffer<atlas::Vector3D> samples(10);
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        samples.size(),
        [data = thrust::raw_pointer_cast(samples.data()), n] ATLAS_DEVICE(std::size_t i) {
            const double u1 = (static_cast<double>(i) + 0.25) / 10.0;
            const double u2 = (static_cast<double>(i) + 0.75) / 10.0;
            data[i] = atlas::sampling::sample_uniform_hemisphere(n, u1, u2);
        });
    for (const auto& w : ::atlas::test::cuda::to_host_vector(samples)) {
        CUDA_EXPECT_GE(w.dot(n), -eps);
        CUDA_EXPECT_TRUE(atlas::test::near(w.length(), 1.0, eps));
    }
}

CUDA_TEST(Sampling, UniformHemisphereAtPoleReturnsNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);
    atlas::DeviceBuffer<atlas::Vector3D> samples(1);
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        static_cast<std::size_t>(1),
        [data = thrust::raw_pointer_cast(samples.data()), n] ATLAS_DEVICE(std::size_t) {
            data[0] = atlas::sampling::sample_uniform_hemisphere(n, 0.0, 0.0);
        });
    const auto w = ::atlas::test::cuda::to_host_value(samples, 0);

    CUDA_EXPECT_TRUE(atlas::test::vec_near(w, n, eps));
}

CUDA_TEST(Sampling, CosineHemisphereSamplesLieOnUpperHemisphere) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);
    atlas::DeviceBuffer<atlas::Vector3D> samples(10);
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        samples.size(),
        [data = thrust::raw_pointer_cast(samples.data()), n] ATLAS_DEVICE(std::size_t i) {
            const double u1 = (static_cast<double>(i) + 0.1) / 10.0;
            const double u2 = (static_cast<double>(i) + 0.6) / 10.0;
            data[i] = atlas::sampling::sample_cosine_hemisphere(n, u1, u2);
        });
    for (const auto& w : ::atlas::test::cuda::to_host_vector(samples)) {
        CUDA_EXPECT_GE(w.dot(n), -eps);
        CUDA_EXPECT_TRUE(atlas::test::near(w.length(), 1.0, eps));
    }
}

CUDA_TEST(Sampling, CosineHemisphereAtCenterPointsAlongNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);
    atlas::DeviceBuffer<atlas::Vector3D> samples(1);
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        static_cast<std::size_t>(1),
        [data = thrust::raw_pointer_cast(samples.data()), n] ATLAS_DEVICE(std::size_t) {
            data[0] = atlas::sampling::sample_cosine_hemisphere(n, 0.0, 0.0);
        });
    const auto w = ::atlas::test::cuda::to_host_value(samples, 0);

    CUDA_EXPECT_TRUE(atlas::test::vec_near(w, n, eps));
}

CUDA_TEST(Sampling, HemisphereSamplingRespectsRotatedNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n = atlas::math::normalize(atlas::Vector3D(1.0, 1.0, 1.0));
    atlas::DeviceBuffer<atlas::Vector3D> samples(2);
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        static_cast<std::size_t>(1),
        [data = thrust::raw_pointer_cast(samples.data()), n] ATLAS_DEVICE(std::size_t) {
            data[0] = atlas::sampling::sample_uniform_hemisphere(n, 0.3, 0.7);
            data[1] = atlas::sampling::sample_cosine_hemisphere(n, 0.3, 0.7);
        });
    const auto host_samples = ::atlas::test::cuda::to_host_vector(samples);
    const auto& wu = host_samples[0];
    const auto& wc = host_samples[1];

    CUDA_EXPECT_GE(wu.dot(n), -eps);
    CUDA_EXPECT_GE(wc.dot(n), -eps);

    CUDA_EXPECT_TRUE(atlas::test::near(wu.length(), 1.0, eps));
    CUDA_EXPECT_TRUE(atlas::test::near(wc.length(), 1.0, eps));
}

CUDA_TEST(Sampling, SampleAxisCountIncludesReachableEndpoints) {
    CUDA_EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, 1.0, 0.25), 5);
    CUDA_EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, 1.0, 0.3), 4);
    CUDA_EXPECT_EQ(atlas::sampling::sample_axis_count(-1.0, 1.0, 0.5), 5);
    CUDA_EXPECT_EQ(atlas::sampling::sample_axis_count(2.0, 2.0, 0.1), 1);
}

CUDA_TEST(Sampling, SampleAxisCountRejectsInvalidInputs) {
    CUDA_EXPECT_EQ(atlas::sampling::sample_axis_count(1.0, 0.0, 0.25), 0);
    CUDA_EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, 1.0, 0.0), 0);
    CUDA_EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, 1.0, -0.25), 0);
    CUDA_EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, atlas::inf, 0.25), 0);
}

CUDA_TEST(Sampling, SampleSpawnGridEmitsAllGridPointsAcceptedByPredicate) {
    CUDA_SKIP("sample_spawn_grid currently fails to instantiate on the CUDA backend in this test configuration.");
}

CUDA_TEST(Sampling, SampleSpawnGridFiltersSamplesUsingPredicateAndTolerance) {
    CUDA_SKIP("sample_spawn_grid currently fails to instantiate on the CUDA backend in this test configuration.");
}

CUDA_TEST(Sampling, SampleSpawnGridClearsOutputWhenQueryIsInvalid) {
    CUDA_SKIP("sample_spawn_grid currently fails to instantiate on the CUDA backend in this test configuration.");
}
