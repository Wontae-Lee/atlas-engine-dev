#include "../../utilities/test_utils.h"

#include <atlas/solver/sph/cubic_spline_sph_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::CubicSplineSphKernel;
using atlas::Vector3F;
using atlas::pi;
using atlas::test::vec_near;
using atlas::tol;

constexpr float kernel_tol = 1.0e-5f;

} // namespace

TEST(CubicSplineSphKernel, DensityWeightUsesInnerAndOuterBranches) {
    // Assert: the cubic spline kernel evaluates both compact-support branches.
    EXPECT_NEAR(CubicSplineSphKernel<float>::density_weight(0.0f, 1.0f),
                1.0f / static_cast<float>(pi),
                kernel_tol);
    EXPECT_NEAR(CubicSplineSphKernel<float>::density_weight(0.5f, 1.0f),
                0.25f / static_cast<float>(pi),
                kernel_tol);
    EXPECT_NEAR(CubicSplineSphKernel<float>::density_weight(1.0f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(CubicSplineSphKernel<float>::density_weight(1.1f, 1.0f), 0.0f, tol);
}

TEST(CubicSplineSphKernel, GradientAndLaplacianFollowBranchFormulas) {
    // Arrange: select one point from the inner cubic branch.
    const Vector3F delta(0.25f, 0.0f, 0.0f);

    // Act: evaluate kernel derivative terms.
    const Vector3F gradient = CubicSplineSphKernel<float>::pressure_gradient(delta, 0.25f, 1.0f);
    const float laplacian = CubicSplineSphKernel<float>::viscosity_laplacian(0.25f, 1.0f);

    // Assert: analytic values match the implementation formulas.
    EXPECT_TRUE(vec_near(
        gradient,
        Vector3F(-1.875f / static_cast<float>(pi), 0.0f, 0.0f),
        kernel_tol));
    EXPECT_NEAR(laplacian, -3.0f / static_cast<float>(pi), kernel_tol);
    EXPECT_TRUE(vec_near(
        CubicSplineSphKernel<float>::pressure_gradient(delta, 0.0f, 1.0f),
        Vector3F(0.0f, 0.0f, 0.0f),
        tol));
}
