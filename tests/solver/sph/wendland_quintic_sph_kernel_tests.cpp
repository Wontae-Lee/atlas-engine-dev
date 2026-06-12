#include "../../utilities/test_utils.h"

#include <atlas/solver/sph/wendland_quintic_sph_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::Vector3F;
using atlas::WendlandQuinticSphKernel;
using atlas::pi;
using atlas::test::vec_near;
using atlas::tol;

constexpr float kernel_tol = 1.0e-5f;

} // namespace

TEST(WendlandQuinticSphKernel, DensityWeightUsesCompactSupport) {
    // Assert: the Wendland kernel has a finite center value and vanishes at support.
    EXPECT_NEAR(WendlandQuinticSphKernel<float>::density_weight(0.0f, 1.0f),
                10.5f / static_cast<float>(pi),
                kernel_tol);
    EXPECT_NEAR(WendlandQuinticSphKernel<float>::density_weight(1.0f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(WendlandQuinticSphKernel<float>::density_weight(1.1f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(WendlandQuinticSphKernel<float>::density_weight(0.5f, 0.0f), 0.0f, tol);
}

TEST(WendlandQuinticSphKernel, GradientAndLaplacianFollowKernelFormula) {
    // Arrange: sample the midpoint of compact support.
    const Vector3F delta(0.5f, 0.0f, 0.0f);

    // Act: evaluate gradient and Laplacian terms.
    const Vector3F gradient = WendlandQuinticSphKernel<float>::pressure_gradient(delta, 0.5f, 1.0f);
    const float laplacian = WendlandQuinticSphKernel<float>::viscosity_laplacian(0.5f, 1.0f);

    // Assert: analytic values match the implementation formulas.
    EXPECT_TRUE(vec_near(
        gradient,
        Vector3F(-13.125f / static_cast<float>(pi), 0.0f, 0.0f),
        kernel_tol));
    EXPECT_NEAR(laplacian, -52.5f / static_cast<float>(pi), kernel_tol);
    EXPECT_TRUE(vec_near(
        WendlandQuinticSphKernel<float>::pressure_gradient(delta, 0.0f, 1.0f),
        Vector3F(0.0f, 0.0f, 0.0f),
        tol));
}
