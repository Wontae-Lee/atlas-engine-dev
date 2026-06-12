#include "../../utilities/test_utils.h"

#include <atlas/solver/sph/sph_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::CubicSplineSphKernel;
using atlas::SphKernel;
using atlas::SphKernelType;
using atlas::StandardSphKernel;
using atlas::Vector3F;
using atlas::WendlandQuinticSphKernel;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(SphKernel, DefaultConstructorSelectsStandardKernel) {
    // Arrange and act: create the runtime SPH kernel wrapper.
    const SphKernel<float> kernel;

    // Assert: default construction selects the standard kernel.
    EXPECT_EQ(kernel.type, SphKernelType::standard);
}

TEST(SphKernel, StaticDispatchMatchesConcreteKernels) {
    // Arrange: use a representative sample inside support.
    const Vector3F delta(0.25f, 0.0f, 0.0f);

    // Assert: runtime dispatch forwards to each concrete kernel implementation.
    EXPECT_NEAR(
        SphKernel<float>::density_weight(SphKernelType::standard, 0.25f, 1.0f),
        StandardSphKernel<float>::density_weight(0.25f, 1.0f),
        tol);
    EXPECT_TRUE(vec_near(
        SphKernel<float>::pressure_gradient(SphKernelType::cubic_spline, delta, 0.25f, 1.0f),
        CubicSplineSphKernel<float>::pressure_gradient(delta, 0.25f, 1.0f),
        tol));
    EXPECT_NEAR(
        SphKernel<float>::viscosity_laplacian(SphKernelType::wendland_quintic, 0.25f, 1.0f),
        WendlandQuinticSphKernel<float>::viscosity_laplacian(0.25f, 1.0f),
        tol);
}

TEST(SphKernel, InstanceDispatchAndCopyPreserveActiveKernel) {
    // Arrange: create a wrapper with a non-default kernel type.
    SphKernel<float> kernel(SphKernelType::cubic_spline);
    SphKernel<float> copied;

    // Act: copy the active kernel.
    copied = kernel;

    // Assert: copied wrappers keep the active tag and dispatch behavior.
    EXPECT_EQ(copied.type, SphKernelType::cubic_spline);
    EXPECT_NEAR(
        copied.density_weight(0.25f, 1.0f),
        CubicSplineSphKernel<float>::density_weight(0.25f, 1.0f),
        tol);
}
