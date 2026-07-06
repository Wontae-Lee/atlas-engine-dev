#include <atlas/solver/sph/standard_sph_kernel.h>

#include <gtest/gtest.h>

namespace {

using atlas::pi;
using atlas::StandardSphKernel;
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

TEST(StandardSphKernel, DensityWeightUsesCompactPoly6Support) {
    EXPECT_NEAR(StandardSphKernel::density_weight(0.0f, 1.0f),
                315.0f / (64.0f * pi),
                kernel_tol);
    EXPECT_NEAR(StandardSphKernel::density_weight(1.0f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(StandardSphKernel::density_weight(1.1f, 1.0f), 0.0f, tol);
    EXPECT_NEAR(StandardSphKernel::density_weight(0.5f, 0.0f), 0.0f, tol);
}

TEST(StandardSphKernel, PressureGradientAndViscosityLaplacianUseSupportRadius) {
    const Vector3 delta(0.5f, 0.0f, 0.0f);

    const Vector3 gradient = StandardSphKernel::pressure_gradient(delta, 0.5f, 1.0f);
    const float laplacian = StandardSphKernel::viscosity_laplacian(0.5f, 1.0f);

    expect_vec_near(
        gradient,
        Vector3(-11.25f / pi, 0.0f, 0.0f),
        kernel_tol);
    EXPECT_NEAR(laplacian, 22.5f / pi, kernel_tol);
    expect_vec_near(
        StandardSphKernel::pressure_gradient(delta, 0.0f, 1.0f),
        Vector3(0.0f, 0.0f, 0.0f),
        tol);
}
