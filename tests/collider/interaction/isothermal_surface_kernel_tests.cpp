#include "../../utilities/test_utils.h"

#include <atlas/collider/interaction/isothermal_surface_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::DiffuseSampling;
using atlas::IsothermalSurfaceInteraction;
using atlas::Vector3F;
using atlas::test::is_finite_vec;
using atlas::tol;

} // namespace

TEST(IsothermalSurfaceKernel, DiffuseSamplingModesAreDistinct) {
    // Assert: the public sampling tags remain distinguishable for dispatch.
    EXPECT_NE(DiffuseSampling::CosineWeighted, DiffuseSampling::Uniform);
}

TEST(IsothermalSurfaceKernel, HeaderExportsUsableInteractionKernel) {
    // Arrange: include this kernel header directly and configure the interaction.
    const auto kernel = IsothermalSurfaceInteraction<float>::builder()
                            .with_diffuse_sampling(DiffuseSampling::Uniform)
                            .with_restitution(0.25f)
                            .with_momentum_acc(0.0f)
                            .build();

    // Act: evaluate a deterministic specular reflection through the exported type.
    const Vector3F incident(0.0f, -4.0f, 0.0f);
    const Vector3F reflected = kernel(incident, Vector3F(0.0f, 1.0f, 0.0f));

    // Assert: direct header use produces finite velocity scaled by restitution.
    EXPECT_TRUE(is_finite_vec(reflected));
    EXPECT_NEAR(reflected.x, 0.0f, tol);
    EXPECT_NEAR(reflected.y, 1.0f, tol);
    EXPECT_NEAR(reflected.z, 0.0f, tol);
}
