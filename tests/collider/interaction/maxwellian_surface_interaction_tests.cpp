#include <atlas/collider/interaction/maxwellian_surface_interaction.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {

using atlas::FluidInternalEnergy;
using atlas::MaterialProperties;
using atlas::MaxwellianInternalEnergyStyle;
using atlas::MaxwellianSurfaceInteraction;
using atlas::Vector3;
using atlas::boltzmann_constant;
using atlas::reflected;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

bool
is_finite_vec(const Vector3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

}

TEST(MaxwellianSurfaceInteraction, DefaultStateIsWellDefined) {
    const MaxwellianSurfaceInteraction interaction;

    EXPECT_NEAR(interaction.temperature(), 273.15f, tol);
    EXPECT_NEAR(interaction.molecular_mass(), 1.0f, tol);
    EXPECT_NEAR(interaction.momentum_acc(), 1.0f, tol);
    EXPECT_NEAR(interaction.trans_acc(), 1.0f, tol);
    EXPECT_NEAR(interaction.rot_acc(), 1.0f, tol);
    EXPECT_NEAR(interaction.vib_acc(), 1.0f, tol);
}

TEST(MaxwellianSurfaceInteraction, BuilderConstructsConfiguredInteraction) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_temperature(300.0f)
                                 .with_molecular_mass(2.0f)
                                 .with_momentum_acc(0.25f)
                                 .with_trans_acc(0.5f)
                                 .with_rot_acc(0.75f)
                                 .with_vib_acc(1.0f)
                                 .with_rot_style(MaxwellianInternalEnergyStyle::discrete)
                                 .with_vib_style(MaxwellianInternalEnergyStyle::none)
                                 .build();

    EXPECT_NEAR(interaction.temperature(), 300.0f, tol);
    EXPECT_NEAR(interaction.molecular_mass(), 2.0f, tol);
    EXPECT_NEAR(interaction.momentum_acc(), 0.25f, tol);
    EXPECT_NEAR(interaction.trans_acc(), 0.5f, tol);
    EXPECT_NEAR(interaction.rot_acc(), 0.75f, tol);
    EXPECT_NEAR(interaction.vib_acc(), 1.0f, tol);
    EXPECT_EQ(interaction.rot_style(), MaxwellianInternalEnergyStyle::discrete);
    EXPECT_EQ(interaction.vib_style(), MaxwellianInternalEnergyStyle::none);
}

TEST(MaxwellianSurfaceInteraction, BuilderMakesHostSharedInteraction) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_temperature(300.0f)
                                 .make_host_shared();

    ASSERT_NE(interaction, nullptr);
    EXPECT_NEAR(interaction->temperature(), 300.0f, tol);
}

TEST(MaxwellianSurfaceInteraction, SettersUpdateState) {
    MaxwellianSurfaceInteraction interaction;

    interaction.set_temperature(450.0f);
    interaction.set_molecular_mass(3.0f);
    interaction.set_momentum_acc(0.25f);
    interaction.set_trans_acc(0.5f);
    interaction.set_rot_acc(0.75f);
    interaction.set_vib_acc(0.125f);
    interaction.set_rot_style(MaxwellianInternalEnergyStyle::discrete);
    interaction.set_vib_style(MaxwellianInternalEnergyStyle::none);
    interaction.set_accommodation(0.2f, 0.4f, 0.6f, 0.8f);

    EXPECT_NEAR(interaction.temperature(), 450.0f, tol);
    EXPECT_NEAR(interaction.molecular_mass(), 3.0f, tol);
    EXPECT_NEAR(interaction.momentum_acc(), 0.2f, tol);
    EXPECT_NEAR(interaction.trans_acc(), 0.4f, tol);
    EXPECT_NEAR(interaction.rot_acc(), 0.6f, tol);
    EXPECT_NEAR(interaction.vib_acc(), 0.8f, tol);
    EXPECT_EQ(interaction.rot_style(), MaxwellianInternalEnergyStyle::discrete);
    EXPECT_EQ(interaction.vib_style(), MaxwellianInternalEnergyStyle::none);
}

TEST(MaxwellianSurfaceInteraction, BuilderCanSetAllAccommodationCoefficientsTogether) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_accommodation(0.2f, 0.4f, 0.6f, 0.8f)
                                 .build();

    EXPECT_NEAR(interaction.momentum_acc(), 0.2f, tol);
    EXPECT_NEAR(interaction.trans_acc(), 0.4f, tol);
    EXPECT_NEAR(interaction.rot_acc(), 0.6f, tol);
    EXPECT_NEAR(interaction.vib_acc(), 0.8f, tol);
}

TEST(MaxwellianSurfaceInteraction, MostProbableSpeedMatchesKineticTheoryScale) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_temperature(300.0f)
                                 .with_molecular_mass(2.0f)
                                 .build();

    const float expected = std::sqrt(2.0f * boltzmann_constant * 300.0f / 2.0f);
    EXPECT_NEAR(interaction.most_probable_speed(), expected, expected * 1.0e-5f);
}

TEST(MaxwellianSurfaceInteraction, BuilderRejectsInvalidAccommodationCoefficients) {
    EXPECT_THROW(
        static_cast<void>(MaxwellianSurfaceInteraction::builder()
                              .with_temperature(0.0f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(MaxwellianSurfaceInteraction::builder()
                              .with_molecular_mass(0.0f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(MaxwellianSurfaceInteraction::builder()
                              .with_momentum_acc(-0.1f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(MaxwellianSurfaceInteraction::builder()
                              .with_trans_acc(1.1f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(MaxwellianSurfaceInteraction::builder()
                              .with_rot_acc(-0.1f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(MaxwellianSurfaceInteraction::builder()
                              .with_vib_acc(1.1f)
                              .build()),
        std::runtime_error);
}

TEST(MaxwellianSurfaceInteraction, MomentumAccommodationZeroUsesSpecularReflection) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_momentum_acc(0.0f)
                                 .build();

    const Vector3 incident(1.0f, -2.0f, 0.5f);
    const Vector3 normal(0.0f, 1.0f, 0.0f);

    const Vector3 out = interaction.sample(
        incident,
        normal,
        1.0f,
        0.5f,
        0.25f,
        0.75f,
        Vector3(1.0f, 0.0f, 0.0f));

    expect_vec_near(out, reflected(incident, normal));
}

TEST(MaxwellianSurfaceInteraction, MomentumAccommodationOneUsesDiffuseMaxwellianSampling) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_temperature(300.0f)
                                 .with_molecular_mass(2.0f)
                                 .with_momentum_acc(1.0f)
                                 .build();

    const Vector3 out = interaction.sample(
        Vector3(1.0f, -2.0f, 0.5f),
        Vector3(0.0f, 1.0f, 0.0f),
        1.0f,
        0.5f,
        0.25f,
        0.75f,
        Vector3(1.0f, 0.0f, 0.0f));

    EXPECT_TRUE(is_finite_vec(out));
    EXPECT_GT(out.length(), 0.0f);
}

TEST(MaxwellianSurfaceInteraction, OperatorProducesFiniteOutgoingVelocity) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_momentum_acc(1.0f)
                                 .build();

    const auto out = interaction(
        Vector3(1.0f, -2.0f, 0.5f),
        Vector3(0.0f, 1.0f, 0.0f));

    EXPECT_TRUE(is_finite_vec(out));
    EXPECT_GT(out.length(), 0.0f);
}

TEST(MaxwellianSurfaceInteraction, InternalEnergyAccommodationZeroPreservesIncidentEnergy) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_accommodation(1.0f, 0.0f, 0.0f, 0.0f)
                                 .build();

    const FluidInternalEnergy incident { 1.0f, 2.0f, 3.0f };
    const auto out = interaction.sample_internal_energy(
        incident,
        0.25f,
        0.5f,
        0.25f,
        0.5f,
        0.25f,
        0.5f);

    EXPECT_NEAR(out.translational, incident.translational, tol);
    EXPECT_NEAR(out.rotational, incident.rotational, tol);
    EXPECT_NEAR(out.vibrational, incident.vibrational, tol);
}

TEST(MaxwellianSurfaceInteraction, InternalEnergyAccommodationOneSamplesWallEnergy) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_temperature(300.0f)
                                 .with_accommodation(1.0f, 1.0f, 1.0f, 1.0f)
                                 .build();

    const FluidInternalEnergy incident { 1.0f, 2.0f, 3.0f };
    const auto out = interaction.sample_internal_energy(
        incident,
        0.25f,
        0.5f,
        0.5f,
        0.5f,
        0.75f,
        0.5f);

    const float wall_energy = boltzmann_constant * 300.0f;
    EXPECT_NEAR(out.translational, wall_energy * -std::log(0.25f), wall_energy * 1.0e-5f);
    EXPECT_NEAR(out.rotational, wall_energy * -std::log(0.5f), wall_energy * 1.0e-5f);
    EXPECT_NEAR(out.vibrational, wall_energy * -std::log(0.75f), wall_energy * 1.0e-5f);
}

TEST(MaxwellianSurfaceInteraction, InternalEnergyUsesDeterministicSamples) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_accommodation(1.0f, 1.0f, 1.0f, 1.0f)
                                 .build();

    const auto out = interaction.internal_energy(FluidInternalEnergy { 1.0f, 2.0f, 3.0f });

    EXPECT_TRUE(std::isfinite(out.translational));
    EXPECT_TRUE(std::isfinite(out.rotational));
    EXPECT_TRUE(std::isfinite(out.vibrational));
    EXPECT_GE(out.translational, 0.0f);
    EXPECT_GE(out.rotational, 0.0f);
    EXPECT_GE(out.vibrational, 0.0f);
}

TEST(MaxwellianSurfaceInteraction, InternalEnergyModeSamplerPreservesEnergyWhenAccommodationIsZero) {
    const MaxwellianSurfaceInteraction interaction;

    const float out = interaction.sample_internal_energy_mode(
        2.5f,
        0.0f,
        0.25f,
        0.5f);

    EXPECT_NEAR(out, 2.5f, tol);
}

TEST(MaxwellianSurfaceInteraction, InternalEnergyModeSamplerUsesWallEnergyWhenAccommodationIsOne) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_temperature(300.0f)
                                 .build();

    const float out = interaction.sample_internal_energy_mode(
        2.5f,
        1.0f,
        0.25f,
        0.5f);

    const float wall_energy = boltzmann_constant * 300.0f;
    EXPECT_NEAR(out, wall_energy * -std::log(0.25f), wall_energy * 1.0e-5f);
}

TEST(MaxwellianSurfaceInteraction, SpecularSurfaceHitPreservesInternalEnergy) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_accommodation(0.0f, 1.0f, 1.0f, 1.0f)
                                 .build();

    const FluidInternalEnergy incident { 1.0f, 2.0f, 3.0f };
    const auto out = interaction.internal_energy(
        incident,
        Vector3(1.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f));

    EXPECT_NEAR(out.translational, incident.translational, tol);
    EXPECT_NEAR(out.rotational, incident.rotational, tol);
    EXPECT_NEAR(out.vibrational, incident.vibrational, tol);
}

TEST(MaxwellianSurfaceInteraction, DiffuseSurfaceHitAppliesInternalEnergyAccommodation) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_accommodation(1.0f, 1.0f, 1.0f, 1.0f)
                                 .build();

    const FluidInternalEnergy incident { 1.0f, 2.0f, 3.0f };
    const auto out = interaction.internal_energy(
        incident,
        Vector3(1.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f));

    EXPECT_TRUE(std::isfinite(out.translational));
    EXPECT_TRUE(std::isfinite(out.rotational));
    EXPECT_TRUE(std::isfinite(out.vibrational));
    EXPECT_GE(out.translational, 0.0f);
    EXPECT_GE(out.rotational, 0.0f);
    EXPECT_GE(out.vibrational, 0.0f);
}

TEST(MaxwellianSurfaceInteraction, DiffuseSurfaceHitSamplesSpartaInternalEnergyFromMaterial) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_accommodation(1.0f, 1.0f, 1.0f, 1.0f)
                                 .build();
    const auto material = MaterialProperties::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .with_rotational_dof(2)
                              .with_vibrational_dof(2)
                              .build();

    const FluidInternalEnergy incident { 1.0f, 2.0f, 3.0f };
    const auto out = interaction.internal_energy(
        incident,
        Vector3(1.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f),
        material);

    EXPECT_NEAR(out.translational, incident.translational, tol);
    EXPECT_GE(out.rotational, 0.0f);
    EXPECT_GE(out.vibrational, 0.0f);
    EXPECT_NE(out.rotational, incident.rotational);
    EXPECT_NE(out.vibrational, incident.vibrational);
}

TEST(MaxwellianSurfaceInteraction, DiffuseSurfaceHitClearsDisabledInternalEnergyModes) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_accommodation(1.0f, 1.0f, 1.0f, 1.0f)
                                 .build();
    const auto material = MaterialProperties::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .build();

    const auto out = interaction.internal_energy(
        FluidInternalEnergy { 1.0f, 2.0f, 3.0f },
        Vector3(1.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f),
        material);

    EXPECT_NEAR(out.rotational, 0.0f, tol);
    EXPECT_NEAR(out.vibrational, 0.0f, tol);
}

TEST(MaxwellianSurfaceInteraction, DiffuseRotationalEnergySamplerUsesMaterialDofAndStyle) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_rot_style(MaxwellianInternalEnergyStyle::smooth)
                                 .build();
    const auto material = MaterialProperties::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .with_rotational_dof(2)
                              .build();

    const float out = interaction.sample_diffuse_rotational_energy(
        material,
        Vector3(1.0f, 2.0f, 3.0f));

    EXPECT_TRUE(std::isfinite(out));
    EXPECT_GE(out, 0.0f);
}

TEST(MaxwellianSurfaceInteraction, DiffuseRotationalEnergySamplerHandlesDiscreteAndDisabledModes) {
    const auto discrete = MaxwellianSurfaceInteraction::builder()
                              .with_rot_style(MaxwellianInternalEnergyStyle::discrete)
                              .build();
    const auto disabled = MaxwellianSurfaceInteraction::builder()
                              .with_rot_style(MaxwellianInternalEnergyStyle::none)
                              .build();
    const auto material = MaterialProperties::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .with_rotational_dof(2)
                              .with_rotational_temperature(100.0f)
                              .build();

    const float discrete_energy = discrete.sample_diffuse_rotational_energy(material, Vector3(1.0f, 2.0f, 3.0f));
    const float disabled_energy = disabled.sample_diffuse_rotational_energy(material, Vector3(1.0f, 2.0f, 3.0f));

    EXPECT_TRUE(std::isfinite(discrete_energy));
    EXPECT_GE(discrete_energy, 0.0f);
    EXPECT_NEAR(disabled_energy, 0.0f, tol);
}

TEST(MaxwellianSurfaceInteraction, DiffuseVibrationalEnergySamplerUsesMaterialDofAndStyle) {
    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_vib_style(MaxwellianInternalEnergyStyle::smooth)
                                 .build();
    const auto material = MaterialProperties::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .with_vibrational_dof(2)
                              .build();

    const float out = interaction.sample_diffuse_vibrational_energy(
        material,
        Vector3(1.0f, 2.0f, 3.0f));

    EXPECT_TRUE(std::isfinite(out));
    EXPECT_GE(out, 0.0f);
}

TEST(MaxwellianSurfaceInteraction, DiffuseVibrationalEnergySamplerHandlesDiscreteAndDisabledModes) {
    const auto discrete = MaxwellianSurfaceInteraction::builder()
                              .with_vib_style(MaxwellianInternalEnergyStyle::discrete)
                              .build();
    const auto disabled = MaxwellianSurfaceInteraction::builder()
                              .with_vib_style(MaxwellianInternalEnergyStyle::none)
                              .build();
    const auto material = MaterialProperties::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .with_vibrational_dof(2)
                              .with_characteristic_vibrational_temperature(100.0f)
                              .build();

    const float discrete_energy = discrete.sample_diffuse_vibrational_energy(material, Vector3(1.0f, 2.0f, 3.0f));
    const float disabled_energy = disabled.sample_diffuse_vibrational_energy(material, Vector3(1.0f, 2.0f, 3.0f));

    EXPECT_TRUE(std::isfinite(discrete_energy));
    EXPECT_GE(discrete_energy, 0.0f);
    EXPECT_NEAR(disabled_energy, 0.0f, tol);
}

TEST(MaxwellianSurfaceInteraction, DiffuseSmoothEnergySamplerHandlesHigherDof) {
    const MaxwellianSurfaceInteraction interaction;

    const float out = interaction.sample_diffuse_smooth_energy(
        4,
        Vector3(1.0f, 2.0f, 3.0f),
        0.5f);

    EXPECT_TRUE(std::isfinite(out));
    EXPECT_GE(out, 0.0f);
}
