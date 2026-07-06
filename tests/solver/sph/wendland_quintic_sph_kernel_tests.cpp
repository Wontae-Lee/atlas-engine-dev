#include <atlas/solver/sph/wendland_quintic_sph_kernel.h>

#include <gtest/gtest.h>

namespace {

using atlas::pi;
using atlas::tol;
using atlas::Vector3;
using atlas::WendlandQuinticSphKernel;

constexpr float kernel_tol = 1.0e-5f;

void
expect_vec_near(const Vector3& actual, const Vector3& expected, const float tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

}

TEST(WendlandQuinticSphKernel, DensityWeightUsesCompactSupport) {
    EXPECT_NEAR(WendlandQuinticSphKernel::density_weight(0.0f, 1.0f),
                10.5f / pi,
                kernel_tol);
    EXPECT_NEAR(WendlandQuinticSphKernel::density_weight(1.0f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(WendlandQuinticSphKernel::density_weight(1.1f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(WendlandQuinticSphKernel::density_weight(0.5f, 0.0f), 0.0f, tol);
}

TEST(WendlandQuinticSphKernel, GradientAndLaplacianFollowKernelFormula) {
    const Vector3 delta(0.5f, 0.0f, 0.0f);

    const Vector3 gradient = WendlandQuinticSphKernel::pressure_gradient(delta, 0.5f, 1.0f);
    const float laplacian = WendlandQuinticSphKernel::viscosity_laplacian(0.5f, 1.0f);

    expect_vec_near(
        gradient,
        Vector3(-13.125f / pi, 0.0f, 0.0f),
        kernel_tol);
    EXPECT_NEAR(laplacian, -52.5f / pi, kernel_tol);
    expect_vec_near(
        WendlandQuinticSphKernel::pressure_gradient(delta, 0.0f, 1.0f),
        Vector3(0.0f, 0.0f, 0.0f),
        tol);
}
