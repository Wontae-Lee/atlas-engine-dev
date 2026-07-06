#include <atlas/solver/sph/cubic_spline_sph_kernel.h>

#include <gtest/gtest.h>

namespace {

using atlas::CubicSplineSphKernel;
using atlas::pi;
using atlas::tol;
using atlas::Vector3;

constexpr float kernel_tol = 1.0e-5f;

void
expect_vec_near(const Vector3& actual, const Vector3& expected, const float tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

}

TEST(CubicSplineSphKernel, DensityWeightUsesInnerAndOuterBranches) {
    EXPECT_NEAR(CubicSplineSphKernel::density_weight(0.0f, 1.0f),
                1.0f / pi,
                kernel_tol);
    EXPECT_NEAR(CubicSplineSphKernel::density_weight(0.5f, 1.0f),
                0.25f / pi,
                kernel_tol);
    EXPECT_NEAR(CubicSplineSphKernel::density_weight(1.0f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(CubicSplineSphKernel::density_weight(1.1f, 1.0f), 0.0f, tol);
}

TEST(CubicSplineSphKernel, GradientAndLaplacianFollowBranchFormulas) {
    const Vector3 delta(0.25f, 0.0f, 0.0f);

    const Vector3 gradient = CubicSplineSphKernel::pressure_gradient(delta, 0.25f, 1.0f);
    const float laplacian = CubicSplineSphKernel::viscosity_laplacian(0.25f, 1.0f);

    expect_vec_near(
        gradient,
        Vector3(-1.875f / pi, 0.0f, 0.0f),
        kernel_tol);
    EXPECT_NEAR(laplacian, -3.0f / pi, kernel_tol);
    expect_vec_near(
        CubicSplineSphKernel::pressure_gradient(delta, 0.0f, 1.0f),
        Vector3(0.0f, 0.0f, 0.0f),
        tol);
}
