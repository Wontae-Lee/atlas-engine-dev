#include "../../utilities/test_utils.h"

#include <atlas/solver/sph/standard_sph_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::StandardSphKernel;
using atlas::Vector3F;
using atlas::pi;
using atlas::test::vec_near;
using atlas::tol;

constexpr float kernel_tol = 1.0e-5f;

} // namespace

TEST(StandardSphKernel, DensityWeightUsesCompactPoly6Support) {
    // Assert: the density kernel is positive inside support and zero outside it.
    EXPECT_NEAR(StandardSphKernel<float>::density_weight(0.0f, 1.0f),
                315.0f / (64.0f * static_cast<float>(pi)),
                kernel_tol);
    EXPECT_NEAR(StandardSphKernel<float>::density_weight(1.0f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(StandardSphKernel<float>::density_weight(1.1f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(StandardSphKernel<float>::density_weight(0.5f, 0.0f), 0.0f, tol);
}

TEST(StandardSphKernel, PressureGradientAndViscosityLaplacianUseSupportRadius) {
    // Arrange: sample the kernel halfway through its compact support.
    const Vector3F delta(0.5f, 0.0f, 0.0f);

    // Act: evaluate gradient and Laplacian terms.
    const Vector3F gradient = StandardSphKernel<float>::pressure_gradient(delta, 0.5f, 1.0f);
    const float laplacian = StandardSphKernel<float>::viscosity_laplacian(0.5f, 1.0f);

    // Assert: analytic values match the implementation formulas.
    EXPECT_TRUE(vec_near(
        gradient,
        Vector3F(-11.25f / static_cast<float>(pi), 0.0f, 0.0f),
        kernel_tol));
    EXPECT_NEAR(laplacian, 22.5f / static_cast<float>(pi), kernel_tol);
    EXPECT_TRUE(vec_near(
        StandardSphKernel<float>::pressure_gradient(delta, 0.0f, 1.0f),
        Vector3F(0.0f, 0.0f, 0.0f),
        tol));
}
