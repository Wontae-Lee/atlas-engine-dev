#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>

#include <cmath>
#include <gtest/gtest.h>

TEST(Sampling, BuildOrthonormalBasisProducesOrthogonalFrame) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n = atlas::math::normalize(atlas::Vector3D(0.3, 0.4, 0.5));

    atlas::Vector3D t, b;
    atlas::sampling::build_orthonormal_basis(n, t, b);

    EXPECT_TRUE(atlas::test::near(t.length(), 1.0, eps));
    EXPECT_TRUE(atlas::test::near(b.length(), 1.0, eps));

    EXPECT_TRUE(atlas::test::near(n.dot(t), 0.0, eps));
    EXPECT_TRUE(atlas::test::near(n.dot(b), 0.0, eps));
    EXPECT_TRUE(atlas::test::near(t.dot(b), 0.0, eps));

    const auto bx = atlas::math::cross(n, t);
    EXPECT_TRUE(atlas::test::vec_near(b, bx, eps));
}

TEST(Sampling, BuildOrthonormalBasisHandlesAxisAlignedNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);

    atlas::Vector3D t, b;
    atlas::sampling::build_orthonormal_basis(n, t, b);

    EXPECT_TRUE(atlas::test::near(n.dot(t), 0.0, eps));
    EXPECT_TRUE(atlas::test::near(n.dot(b), 0.0, eps));
    EXPECT_TRUE(atlas::test::near(t.dot(b), 0.0, eps));
}

TEST(Sampling, UniformHemisphereSamplesLieOnUpperHemisphere) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);

    for (int i = 0; i < 10; ++i) {
        const double u1 = (i + 0.25) / 10.0;
        const double u2 = (i + 0.75) / 10.0;

        const auto w = atlas::sampling::sample_uniform_hemisphere(n, u1, u2);

        EXPECT_GE(w.dot(n), -eps);

        EXPECT_TRUE(atlas::test::near(w.length(), 1.0, eps));
    }
}

TEST(Sampling, UniformHemisphereAtPoleReturnsNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);

    const auto w = atlas::sampling::sample_uniform_hemisphere(n, 0.0, 0.0);

    EXPECT_TRUE(atlas::test::vec_near(w, n, eps));
}

TEST(Sampling, CosineHemisphereSamplesLieOnUpperHemisphere) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);

    for (int i = 0; i < 10; ++i) {
        const double u1 = (i + 0.1) / 10.0;
        const double u2 = (i + 0.6) / 10.0;

        const auto w = atlas::sampling::sample_cosine_hemisphere(n, u1, u2);

        EXPECT_GE(w.dot(n), -eps);
        EXPECT_TRUE(atlas::test::near(w.length(), 1.0, eps));
    }
}

TEST(Sampling, CosineHemisphereAtCenterPointsAlongNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n(0.0, 0.0, 1.0);

    const auto w = atlas::sampling::sample_cosine_hemisphere(n, 0.0, 0.0);

    EXPECT_TRUE(atlas::test::vec_near(w, n, eps));
}

TEST(Sampling, HemisphereSamplingRespectsRotatedNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector3D n = atlas::math::normalize(atlas::Vector3D(1.0, 1.0, 1.0));

    const auto wu = atlas::sampling::sample_uniform_hemisphere(n, 0.3, 0.7);
    const auto wc = atlas::sampling::sample_cosine_hemisphere(n, 0.3, 0.7);

    EXPECT_GE(wu.dot(n), -eps);
    EXPECT_GE(wc.dot(n), -eps);

    EXPECT_TRUE(atlas::test::near(wu.length(), 1.0, eps));
    EXPECT_TRUE(atlas::test::near(wc.length(), 1.0, eps));
}

TEST(Sampling, SampleAxisCountIncludesReachableEndpoints) {
    EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, 1.0, 0.25), 5);
    EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, 1.0, 0.3), 4);
    EXPECT_EQ(atlas::sampling::sample_axis_count(-1.0, 1.0, 0.5), 5);
    EXPECT_EQ(atlas::sampling::sample_axis_count(2.0, 2.0, 0.1), 1);
}

TEST(Sampling, SampleAxisCountRejectsInvalidInputs) {
    EXPECT_EQ(atlas::sampling::sample_axis_count(1.0, 0.0, 0.25), 0);
    EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, 1.0, 0.0), 0);
    EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, 1.0, -0.25), 0);
    EXPECT_EQ(atlas::sampling::sample_axis_count(0.0, atlas::inf, 0.25), 0);
}
