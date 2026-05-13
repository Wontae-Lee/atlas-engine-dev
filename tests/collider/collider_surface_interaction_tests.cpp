#include "../utilities/test_utils.h"

#include <atlas/collider/collider_surface_interaction.h>

#include <testkit/testkit.h>

namespace {

using atlas::ColliderSurfaceInteraction;
using atlas::tol;
using atlas::Vector3F;
using atlas::math::reflected;
using atlas::system::DiffuseSampling;
using atlas::test::is_finite_vec;
using atlas::test::vec_near;

} // namespace

TEST(ColliderSurfaceInteraction, DefaultStateIsWellDefined) {
    // Arrange and act: construct the interaction with default settings.
    const ColliderSurfaceInteraction<float> interaction;

    // Assert: defaults are physically valid and deterministic.
    EXPECT_EQ(interaction.diffuse_sampling(), DiffuseSampling::Uniform);
    EXPECT_NEAR(interaction.restitution(), 1.0f, tol);
    EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 1.0f, tol);
    EXPECT_NEAR(interaction.temperature(), 273.15f, tol);
}

TEST(ColliderSurfaceInteraction, SettersUpdateState) {
    // Arrange: create an interaction object with default values.
    ColliderSurfaceInteraction<float> interaction;

    // Act: update all configurable surface-interaction parameters.
    interaction.set_diffuse_sampling(DiffuseSampling::CosineWeighted);
    interaction.set_restitution(0.75f);
    interaction.set_tangential_momentum_accommodation(0.25f);
    interaction.set_temperature(350.0f);

    // Assert: all setter changes are preserved.
    EXPECT_EQ(interaction.diffuse_sampling(), DiffuseSampling::CosineWeighted);
    EXPECT_NEAR(interaction.restitution(), 0.75f, tol);
    EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 0.25f, tol);
    EXPECT_NEAR(interaction.temperature(), 350.0f, tol);
}

TEST(ColliderSurfaceInteraction, BuilderConstructsConfiguredInteraction) {
    // Arrange and act: build an interaction with explicit configuration.
    const auto interaction = ColliderSurfaceInteraction<float>::builder()
                                 .with_diffuse_sampling(DiffuseSampling::CosineWeighted)
                                 .with_restitution(0.5f)
                                 .with_tangential_momentum_accommodation(0.0f)
                                 .with_temperature(400.0f)
                                 .build();

    // Assert: builder-provided values are applied.
    EXPECT_EQ(interaction.diffuse_sampling(), DiffuseSampling::CosineWeighted);
    EXPECT_NEAR(interaction.restitution(), 0.5f, tol);
    EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 0.0f, tol);
    EXPECT_NEAR(interaction.temperature(), 400.0f, tol);
}

TEST(ColliderSurfaceInteraction, BuilderRejectsInvalidParameters) {
    // Assert: restitution must be non-negative.
    EXPECT_THROW(
        ColliderSurfaceInteraction<float>::builder()
            .with_restitution(-1.0f)
            .build(),
        std::runtime_error);

    // Assert: tangential momentum accommodation must stay within the valid range.
    EXPECT_THROW(
        ColliderSurfaceInteraction<float>::builder()
            .with_tangential_momentum_accommodation(2.0f)
            .build(),
        std::runtime_error);

    // Assert: surface temperature must be non-negative.
    EXPECT_THROW(
        ColliderSurfaceInteraction<float>::builder()
            .with_temperature(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(ColliderSurfaceInteraction, SpecularModeMatchesReflectedDirection) {
    // Arrange: disable diffuse accommodation to force specular reflection.
    ColliderSurfaceInteraction<float> interaction;
    interaction.set_restitution(0.5f);
    interaction.set_tangential_momentum_accommodation(0.0f);

    const Vector3F incident(1.0f, -2.0f, 0.0f);
    const Vector3F normal(0.0f, 1.0f, 0.0f);

    // Act: compute the outgoing velocity after surface interaction.
    const Vector3F out = interaction(incident, normal);

    // Assert: pure specular mode matches reflected velocity scaled by restitution.
    const Vector3F expected = reflected(incident, normal) * 0.5f;
    EXPECT_TRUE(vec_near(out, expected, tol));
}

TEST(ColliderSurfaceInteraction, DiffuseModeReturnsFiniteDirection) {
    // Arrange: use full diffuse accommodation with reduced restitution.
    ColliderSurfaceInteraction<float> interaction;
    interaction.set_restitution(0.5f);
    interaction.set_tangential_momentum_accommodation(1.0f);

    const Vector3F incident(1.0f, -1.0f, 0.5f);

    // Act: sample the outgoing diffuse velocity.
    const Vector3F out = interaction(incident, Vector3F(0.0f, 1.0f, 0.0f));

    // Assert: sampled velocity is finite, non-zero, and scaled by restitution.
    EXPECT_TRUE(is_finite_vec(out));
    EXPECT_GT(out.length(), 0.0f);
    EXPECT_NEAR(out.length(), incident.length() * 0.5f, tol);
}