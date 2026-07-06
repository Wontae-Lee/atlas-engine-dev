#include <atlas/solver/sph/sph_kernel.h>

#include <gtest/gtest.h>

namespace {

using atlas::CubicSplineSphKernel;
using atlas::SphKernel;
using atlas::SphKernelType;
using atlas::StandardSphKernel;
using atlas::tol;
using atlas::Float3;
using atlas::WendlandQuinticSphKernel;

void
expect_vec_near(const Float3& actual, const Float3& expected, const float tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

}

TEST(SphKernel, DefaultConstructorSelectsStandardKernel) {
    const SphKernel kernel;

    EXPECT_EQ(kernel.type, SphKernelType::standard);
}

TEST(SphKernel, StaticDispatchMatchesConcreteKernels) {
    const Float3 delta(0.25f, 0.0f, 0.0f);

    EXPECT_NEAR(
        SphKernel::density_weight(SphKernelType::standard, 0.25f, 1.0f),
        StandardSphKernel::density_weight(0.25f, 1.0f),
        tol);
    expect_vec_near(
        SphKernel::pressure_gradient(SphKernelType::cubic_spline, delta, 0.25f, 1.0f),
        CubicSplineSphKernel::pressure_gradient(delta, 0.25f, 1.0f),
        tol);
    EXPECT_NEAR(
        SphKernel::viscosity_laplacian(SphKernelType::wendland_quintic, 0.25f, 1.0f),
        WendlandQuinticSphKernel::viscosity_laplacian(0.25f, 1.0f),
        tol);
}

TEST(SphKernel, InstanceDispatchAndCopyPreserveActiveKernel) {
    SphKernel kernel(SphKernelType::cubic_spline);
    SphKernel copied;

    copied = kernel;

    EXPECT_EQ(copied.type, SphKernelType::cubic_spline);
    EXPECT_NEAR(
        copied.density_weight(0.25f, 1.0f),
        CubicSplineSphKernel::density_weight(0.25f, 1.0f),
        tol);
}
