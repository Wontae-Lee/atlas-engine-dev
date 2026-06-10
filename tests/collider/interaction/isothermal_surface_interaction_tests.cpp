#include "../../utilities/test_utils.h"

#include <atlas/collider/interaction/isothermal_surface_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::IsothermalSurfaceInteraction;
using atlas::tol;
using atlas::Vector3F;
using atlas::math::reflected;
using atlas::system::DiffuseSampling;
using atlas::test::is_finite_vec;
using atlas::test::vec_near;

} // namespace

TEST(IsothermalSurfaceInteraction, DefaultStateIsWellDefined) {
    // Arrange and act: construct the interaction with default settings.
    const IsothermalSurfaceInteraction<float> interaction;

    // Assert: defaults are physically valid and deterministic.
    EXPECT_EQ(interaction.diffuse_sampling(), DiffuseSampling::Uniform);
    EXPECT_NEAR(interaction.restitution(), 1.0f, tol);
    EXPECT_NEAR(interaction.momentum_acc(), 1.0f, tol);
    EXPECT_NEAR(interaction.temperature(), 273.15f, tol);
}

TEST(IsothermalSurfaceInteraction, SettersUpdateState) {
    // Arrange: create an interaction object with default values.
    IsothermalSurfaceInteraction<float> interaction;

    // Act: update all configurable surface-interaction parameters.
    interaction.set_diffuse_sampling(DiffuseSampling::CosineWeighted);
    interaction.set_restitution(0.75f);
    interaction.set_momentum_acc(0.25f);
    interaction.set_temperature(350.0f);

    // Assert: all setter changes are preserved.
    EXPECT_EQ(interaction.diffuse_sampling(), DiffuseSampling::CosineWeighted);
    EXPECT_NEAR(interaction.restitution(), 0.75f, tol);
    EXPECT_NEAR(interaction.momentum_acc(), 0.25f, tol);
    EXPECT_NEAR(interaction.temperature(), 350.0f, tol);
}

TEST(IsothermalSurfaceInteraction, BuilderConstructsConfiguredInteraction) {
    // Arrange and act: build an interaction with explicit configuration.
    const auto interaction = IsothermalSurfaceInteraction<float>::builder()
                                 .with_diffuse_sampling(DiffuseSampling::CosineWeighted)
                                 .with_restitution(0.5f)
                                 .with_momentum_acc(0.0f)
                                 .with_temperature(400.0f)
                                 .build();

    // Assert: builder-provided values are applied.
    EXPECT_EQ(interaction.diffuse_sampling(), DiffuseSampling::CosineWeighted);
    EXPECT_NEAR(interaction.restitution(), 0.5f, tol);
    EXPECT_NEAR(interaction.momentum_acc(), 0.0f, tol);
    EXPECT_NEAR(interaction.temperature(), 400.0f, tol);
}

TEST(IsothermalSurfaceInteraction, BuilderMakesHostSharedInteraction) {
    const auto interaction = IsothermalSurfaceInteraction<float>::builder()
                                 .with_restitution(0.5f)
                                 .make_host_shared();

    ASSERT_NE(interaction, nullptr);
    EXPECT_NEAR(interaction->restitution(), 0.5f, tol);
}

TEST(IsothermalSurfaceInteraction, BuilderRejectsInvalidParameters) {
    // Assert: restitution must be non-negative.
    EXPECT_THROW(
        IsothermalSurfaceInteraction<float>::builder()
            .with_restitution(-1.0f)
            .build(),
        std::runtime_error);

    // Assert: momentum accommodation must stay within the valid range.
    EXPECT_THROW(
        IsothermalSurfaceInteraction<float>::builder()
            .with_momentum_acc(2.0f)
            .build(),
        std::runtime_error);

    // Assert: surface temperature must be non-negative.
    EXPECT_THROW(
        IsothermalSurfaceInteraction<float>::builder()
            .with_temperature(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(IsothermalSurfaceInteraction, SpecularModeMatchesReflectedDirection) {
    // Arrange: disable diffuse accommodation to force specular reflection.
    IsothermalSurfaceInteraction<float> interaction;
    interaction.set_restitution(0.5f);
    interaction.set_momentum_acc(0.0f);

    const Vector3F incident(1.0f, -2.0f, 0.0f);
    const Vector3F normal(0.0f, 1.0f, 0.0f);

    // Act: compute the outgoing velocity after surface interaction.
    const Vector3F out = interaction(incident, normal);

    // Assert: pure specular mode matches reflected velocity scaled by restitution.
    const Vector3F expected = reflected(incident, normal) * 0.5f;
    EXPECT_TRUE(vec_near(out, expected, tol));
}

TEST(IsothermalSurfaceInteraction, DiffuseModeReturnsFiniteDirection) {
    // Arrange: use full diffuse accommodation with reduced restitution.
    IsothermalSurfaceInteraction<float> interaction;
    interaction.set_restitution(0.5f);
    interaction.set_momentum_acc(1.0f);

    const Vector3F incident(1.0f, -1.0f, 0.5f);

    // Act: sample the outgoing diffuse velocity.
    const Vector3F out = interaction(incident, Vector3F(0.0f, 1.0f, 0.0f));

    // Assert: sampled velocity is finite, non-zero, and scaled by restitution.
    EXPECT_TRUE(is_finite_vec(out));
    EXPECT_GT(out.length(), 0.0f);
    EXPECT_NEAR(out.length(), incident.length() * 0.5f, tol);
}
