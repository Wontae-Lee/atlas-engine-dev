#include <atlas/collider/interaction/isothermal_surface_kernel.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {

using atlas::DiffuseSampling;
using atlas::IsothermalSurfaceInteraction;
using atlas::Float3;
using atlas::reflected;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

bool
is_finite_vec(const Float3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

}

TEST(IsothermalSurfaceInteraction, DefaultStateIsWellDefined) {
    const IsothermalSurfaceInteraction interaction;

    EXPECT_EQ(interaction.diffuse_sampling(), DiffuseSampling::uniform);
    EXPECT_NEAR(interaction.restitution(), 1.0f, tol);
    EXPECT_NEAR(interaction.momentum_acc(), 1.0f, tol);
    EXPECT_NEAR(interaction.temperature(), 273.15f, tol);
}

TEST(IsothermalSurfaceInteraction, SettersUpdateState) {
    IsothermalSurfaceInteraction interaction;

    interaction.set_diffuse_sampling(DiffuseSampling::cosine_weighted);
    interaction.set_restitution(0.75f);
    interaction.set_momentum_acc(0.25f);
    interaction.set_temperature(350.0f);

    EXPECT_EQ(interaction.diffuse_sampling(), DiffuseSampling::cosine_weighted);
    EXPECT_NEAR(interaction.restitution(), 0.75f, tol);
    EXPECT_NEAR(interaction.momentum_acc(), 0.25f, tol);
    EXPECT_NEAR(interaction.temperature(), 350.0f, tol);
}

TEST(IsothermalSurfaceInteraction, BuilderConstructsConfiguredInteraction) {
    const auto interaction = IsothermalSurfaceInteraction::builder()
                                 .with_diffuse_sampling(DiffuseSampling::cosine_weighted)
                                 .with_restitution(0.5f)
                                 .with_momentum_acc(0.0f)
                                 .with_temperature(400.0f)
                                 .build();

    EXPECT_EQ(interaction.diffuse_sampling(), DiffuseSampling::cosine_weighted);
    EXPECT_NEAR(interaction.restitution(), 0.5f, tol);
    EXPECT_NEAR(interaction.momentum_acc(), 0.0f, tol);
    EXPECT_NEAR(interaction.temperature(), 400.0f, tol);
}

TEST(IsothermalSurfaceInteraction, BuilderMakesHostSharedInteraction) {
    const auto interaction = IsothermalSurfaceInteraction::builder()
                                 .with_restitution(0.5f)
                                 .make_host_shared();

    ASSERT_NE(interaction, nullptr);
    EXPECT_NEAR(interaction->restitution(), 0.5f, tol);
}

TEST(IsothermalSurfaceInteraction, BuilderRejectsInvalidParameters) {
    EXPECT_THROW(
        static_cast<void>(IsothermalSurfaceInteraction::builder()
                              .with_restitution(-1.0f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(IsothermalSurfaceInteraction::builder()
                              .with_momentum_acc(2.0f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(IsothermalSurfaceInteraction::builder()
                              .with_temperature(-1.0f)
                              .build()),
        std::runtime_error);
}

TEST(IsothermalSurfaceInteraction, SpecularModeMatchesReflectedDirection) {
    IsothermalSurfaceInteraction interaction;
    interaction.set_restitution(0.5f);
    interaction.set_momentum_acc(0.0f);

    const Float3 incident(1.0f, -2.0f, 0.0f);
    const Float3 normal(0.0f, 1.0f, 0.0f);

    const Float3 out = interaction(incident, normal);

    const Float3 expected = reflected(incident, normal) * 0.5f;
    expect_vec_near(out, expected);
}

TEST(IsothermalSurfaceInteraction, DiffuseModeReturnsFiniteDirection) {
    IsothermalSurfaceInteraction interaction;
    interaction.set_restitution(0.5f);
    interaction.set_momentum_acc(1.0f);

    const Float3 incident(1.0f, -1.0f, 0.5f);

    const Float3 out = interaction(incident, Float3(0.0f, 1.0f, 0.0f));

    EXPECT_TRUE(is_finite_vec(out));
    EXPECT_GT(out.length(), 0.0f);
    EXPECT_NEAR(out.length(), incident.length() * 0.5f, tol);
}
