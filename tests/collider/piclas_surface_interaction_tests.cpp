#include "../utilities/test_utils.h"

#include <atlas/collider/piclas_surface_interaction.h>

#include <testkit/testkit.h>

namespace {

using atlas::PiclasSurfaceInteraction;
using atlas::Vector3F;
using atlas::math::reflected;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::test::is_finite_vec;
using atlas::test::vec_near;

constexpr float tol = 1.0e-5f;

} // namespace

TEST(PiclasSurfaceInteraction, BuilderConstructsConfiguredInteraction) {
    const auto particle_properties = MaterialProperties<float>::builder()
                                         .with_type(MaterialType::Molecule)
                                         .with_mass(2.0f)
                                         .with_molecular_mass(2.0f)
                                         .with_characteristic_vibrational_temperature(1000.0f)
                                         .with_max_vibrational_quantum(12)
                                         .with_gamma_quant(0.5f)
                                         .with_interaction_id(20)
                                         .with_fully_ionized(false)
                                         .build();
    const auto wall_properties = MaterialProperties<float>::builder()
                                     .with_type(MaterialType::Solid)
                                     .with_mass(5.0f)
                                     .with_molecular_mass(5.0f)
                                     .with_reference_temperature(400.0f)
                                     .with_rotational_energy(6.0f)
                                     .with_electronic_energy(7.0f)
                                     .build();
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_momentum_accommodation(0.25f)
                                 .with_translational_accommodation(0.5f)
                                 .with_vibrational_accommodation(0.75f)
                                 .with_rotational_accommodation(0.125f)
                                 .with_electronic_accommodation(0.625f)
                                 .with_temperature(400.0f)
                                 .with_molecular_mass(2.0f)
                                 .with_particle_properties(particle_properties)
                                 .with_wall_properties(wall_properties)
                                 .build();

    EXPECT_NEAR(interaction.momentum_accommodation(), 0.25f, tol);
    EXPECT_NEAR(interaction.translational_accommodation(), 0.5f, tol);
    EXPECT_NEAR(interaction.vibrational_accommodation(), 0.75f, tol);
    EXPECT_NEAR(interaction.rotational_accommodation(), 0.125f, tol);
    EXPECT_NEAR(interaction.electronic_accommodation(), 0.625f, tol);
    EXPECT_NEAR(interaction.temperature(), 400.0f, tol);
    EXPECT_NEAR(interaction.molecular_mass(), 2.0f, tol);
    EXPECT_NEAR(interaction.particle_properties().molecular_mass, 2.0f, tol);
    ASSERT_TRUE(interaction.wall_properties().rotational_energy.has_value());
    EXPECT_NEAR(*interaction.wall_properties().rotational_energy, 6.0f, tol);
    EXPECT_TRUE(vec_near(interaction.wall_velocity(), Vector3F(0.0f), tol));
    EXPECT_TRUE(vec_near(interaction.wall_angular_velocity(), Vector3F(0.0f), tol));
    EXPECT_TRUE(vec_near(interaction.wall_rotation_origin(), Vector3F(0.0f), tol));
    EXPECT_FALSE(interaction.only_specular());
    EXPECT_FALSE(interaction.only_diffuse());
    EXPECT_FALSE(interaction.normal_points_out_of_domain());
    EXPECT_TRUE(interaction.use_dsmc());
    EXPECT_EQ(interaction.collision_mode(), 2);
    EXPECT_TRUE(interaction.vibrational_relaxation_enabled());
    EXPECT_TRUE(interaction.electronic_relaxation_enabled());
}

TEST(PiclasSurfaceInteraction, ZeroMomentumAccommodationMatchesSpecularReflection) {
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_momentum_accommodation(0.0f)
                                 .build();
    const Vector3F incident(1.0f, -2.0f, 0.5f);
    const Vector3F normal(0.0f, 1.0f, 0.0f);

    const Vector3F actual = interaction(incident, normal);
    const Vector3F expected = reflected(incident, normal);

    EXPECT_TRUE(vec_near(actual, expected, tol));
}

TEST(PiclasSurfaceInteraction, DiffuseReflectionLeavesSurfaceHemisphere) {
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_momentum_accommodation(1.0f)
                                 .with_translational_accommodation(1.0f)
                                 .with_temperature(300.0f)
                                 .with_molecular_mass(1.0f)
                                 .build();
    const Vector3F incident(1.0f, -2.0f, 0.5f);
    const Vector3F normal(0.0f, 1.0f, 0.0f);

    const Vector3F outgoing = interaction(incident, normal);

    EXPECT_GT(outgoing.dot(normal), 0.0f);
    EXPECT_TRUE(is_finite_vec(outgoing));
}

TEST(PiclasSurfaceInteraction, OnlyDiffuseForcesDiffuseBranch) {
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_momentum_accommodation(0.0f)
                                 .with_only_diffuse(true)
                                 .with_temperature(300.0f)
                                 .with_molecular_mass(1.0f)
                                 .build();
    const Vector3F incident(1.0f, -2.0f, 0.5f);
    const Vector3F normal(0.0f, 1.0f, 0.0f);

    const Vector3F outgoing = interaction(incident, normal);
    const Vector3F specular = reflected(incident, normal);

    EXPECT_GT(outgoing.dot(normal), 0.0f);
    EXPECT_FALSE(vec_near(outgoing, specular, tol));
}

TEST(PiclasSurfaceInteraction, InternalEnergyAccommodationCanSampleWallStates) {
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_vibrational_accommodation(1.0f)
                                 .with_rotational_accommodation(1.0f)
                                 .with_electronic_accommodation(1.0f)
                                 .with_temperature(300.0f)
                                 .build();
    const PiclasSurfaceInteraction<float>::InternalEnergyState old_energy {
        .vibrational = 1.0f,
        .rotational = 2.0f,
        .electronic = 3.0f,
    };
    const PiclasSurfaceInteraction<float>::InternalEnergyParameters parameters {
        .rotational_wall_energy = 4.0f,
        .electronic_wall_energy = 5.0f,
        .characteristic_vibrational_temperature = 1000.0f,
        .gamma_quant = 0.5f,
        .max_vibrational_quantum = 10,
        .enable_vibrational_relaxation = true,
        .enable_electronic_relaxation = true,
    };

    const auto energy = interaction.accommodate_internal_energy(
        old_energy,
        parameters,
        Vector3F(1.0f, 2.0f, 3.0f));

    EXPECT_GT(energy.vibrational, 0.0f);
    EXPECT_NEAR(energy.rotational, 4.0f, tol);
    EXPECT_NEAR(energy.electronic, 5.0f, tol);
}

TEST(PiclasSurfaceInteraction, MaterialPropertiesDriveInternalEnergyAccommodation) {
    const auto particle_properties = MaterialProperties<float>::builder()
                                         .with_type(MaterialType::Molecule)
                                         .with_mass(2.0f)
                                         .with_molecular_mass(2.0f)
                                         .with_characteristic_vibrational_temperature(1000.0f)
                                         .with_max_vibrational_quantum(12)
                                         .with_gamma_quant(0.5f)
                                         .with_interaction_id(2)
                                         .with_fully_ionized(false)
                                         .build();
    const auto wall_properties = MaterialProperties<float>::builder()
                                     .with_type(MaterialType::Solid)
                                     .with_mass(5.0f)
                                     .with_molecular_mass(5.0f)
                                     .with_reference_temperature(300.0f)
                                     .with_rotational_energy(4.0f)
                                     .with_electronic_energy(5.0f)
                                     .build();
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_vibrational_accommodation(1.0f)
                                 .with_rotational_accommodation(1.0f)
                                 .with_electronic_accommodation(1.0f)
                                 .with_particle_properties(particle_properties)
                                 .with_wall_properties(wall_properties)
                                 .build();
    const PiclasSurfaceInteraction<float>::InternalEnergyState old_energy {
        .vibrational = 1.0f,
        .rotational = 2.0f,
        .electronic = 3.0f,
    };

    const auto energy = interaction.accommodate_internal_energy(
        old_energy,
        Vector3F(1.0f, 2.0f, 3.0f));

    EXPECT_GT(energy.vibrational, 0.0f);
    EXPECT_NEAR(energy.rotational, 4.0f, tol);
    EXPECT_NEAR(energy.electronic, 5.0f, tol);
}

TEST(PiclasSurfaceInteraction, DsmcInternalEnergyConditionsCanDisableAccommodation) {
    const auto wall_properties = MaterialProperties<float>::builder()
                                     .with_type(MaterialType::Solid)
                                     .with_mass(5.0f)
                                     .with_molecular_mass(5.0f)
                                     .with_rotational_energy(4.0f)
                                     .with_electronic_energy(5.0f)
                                     .build();
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_vibrational_accommodation(1.0f)
                                 .with_rotational_accommodation(1.0f)
                                 .with_electronic_accommodation(1.0f)
                                 .with_use_dsmc(false)
                                 .with_wall_properties(wall_properties)
                                 .build();
    const PiclasSurfaceInteraction<float>::InternalEnergyState old_energy {
        .vibrational = 1.0f,
        .rotational = 2.0f,
        .electronic = 3.0f,
    };

    const auto energy = interaction.accommodate_internal_energy(
        old_energy,
        Vector3F(1.0f, 2.0f, 3.0f));

    EXPECT_NEAR(energy.vibrational, old_energy.vibrational, tol);
    EXPECT_NEAR(energy.rotational, old_energy.rotational, tol);
    EXPECT_NEAR(energy.electronic, old_energy.electronic, tol);
}

TEST(PiclasSurfaceInteraction, InteractionIdControlsInternalEnergyBranches) {
    const auto electron_properties = MaterialProperties<float>::builder()
                                         .with_type(MaterialType::Ion)
                                         .with_mass(1.0f)
                                         .with_molecular_mass(1.0f)
                                         .with_characteristic_vibrational_temperature(1000.0f)
                                         .with_max_vibrational_quantum(12)
                                         .with_interaction_id(4)
                                         .with_fully_ionized(false)
                                         .build();
    const auto wall_properties = MaterialProperties<float>::builder()
                                     .with_type(MaterialType::Solid)
                                     .with_mass(5.0f)
                                     .with_molecular_mass(5.0f)
                                     .with_rotational_energy(4.0f)
                                     .with_electronic_energy(5.0f)
                                     .build();
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_vibrational_accommodation(1.0f)
                                 .with_rotational_accommodation(1.0f)
                                 .with_electronic_accommodation(1.0f)
                                 .with_particle_properties(electron_properties)
                                 .with_wall_properties(wall_properties)
                                 .build();
    const PiclasSurfaceInteraction<float>::InternalEnergyState old_energy {
        .vibrational = 1.0f,
        .rotational = 2.0f,
        .electronic = 3.0f,
    };

    const auto energy = interaction.accommodate_internal_energy(
        old_energy,
        Vector3F(1.0f, 2.0f, 3.0f));

    EXPECT_NEAR(energy.vibrational, old_energy.vibrational, tol);
    EXPECT_NEAR(energy.rotational, old_energy.rotational, tol);
    EXPECT_NEAR(energy.electronic, old_energy.electronic, tol);
}

TEST(PiclasSurfaceInteraction, CollideUpdatesPositionTrajectoryVelocityAndEnergy) {
    const auto wall_properties = MaterialProperties<float>::builder()
                                     .with_type(MaterialType::Solid)
                                     .with_mass(5.0f)
                                     .with_molecular_mass(5.0f)
                                     .with_rotational_energy(4.0f)
                                     .build();
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_momentum_accommodation(0.0f)
                                 .with_rotational_accommodation(1.0f)
                                 .with_wall_velocity(Vector3F(0.5f, 0.0f, 0.0f))
                                 .with_wall_properties(wall_properties)
                                 .build();
    const PiclasSurfaceInteraction<float>::ParticleState state {
        .position = Vector3F(0.0f, -1.0f, 0.0f),
        .last_position = Vector3F(0.0f, -2.0f, 0.0f),
        .velocity = Vector3F(1.0f, -2.0f, 0.0f),
        .trajectory = Vector3F(0.0f, 1.0f, 0.0f),
        .trajectory_length = 1.0f,
        .internal_energy = {
            .vibrational = 1.0f,
            .rotational = 2.0f,
            .electronic = 3.0f,
        },
    };

    const auto result = interaction.collide(
        state,
        Vector3F(0.0f, 0.0f, 0.0f),
        Vector3F(0.0f, 1.0f, 0.0f),
        0.25f);

    EXPECT_TRUE(vec_near(result.last_position, Vector3F(0.0f, 0.0f, 0.0f), tol));
    EXPECT_TRUE(vec_near(result.velocity, Vector3F(1.0f, 2.0f, 0.0f), tol));
    EXPECT_TRUE(vec_near(result.position, Vector3F(0.25f, 0.5f, 0.0f), tol));
    EXPECT_NEAR(result.trajectory_length, result.position.length(), tol);
    EXPECT_NEAR(result.internal_energy.rotational, 4.0f, tol);
}

TEST(PiclasSurfaceInteraction, RotatingWallContributesPointVelocity) {
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_momentum_accommodation(0.0f)
                                 .with_wall_angular_velocity(Vector3F(0.0f, 0.0f, 2.0f))
                                 .with_wall_rotation_origin(Vector3F(0.0f, 0.0f, 0.0f))
                                 .build();
    const PiclasSurfaceInteraction<float>::ParticleState state {
        .position = Vector3F(1.0f, 0.0f, 0.0f),
        .last_position = Vector3F(1.0f, -1.0f, 0.0f),
        .velocity = Vector3F(0.0f, -1.0f, 0.0f),
    };

    const auto result = interaction.collide(
        state,
        Vector3F(1.0f, 0.0f, 0.0f),
        Vector3F(0.0f, 1.0f, 0.0f),
        0.0f);

    EXPECT_TRUE(vec_near(result.velocity, Vector3F(0.0f, 5.0f, 0.0f), tol));
}

TEST(PiclasSurfaceInteraction, PiclasNormalOrientationSamplesAgainstNormal) {
    const auto interaction = PiclasSurfaceInteraction<float>::builder()
                                 .with_only_diffuse(true)
                                 .with_normal_points_out_of_domain(true)
                                 .with_temperature(300.0f)
                                 .with_molecular_mass(1.0f)
                                 .build();
    const Vector3F incident(1.0f, 2.0f, 0.5f);
    const Vector3F outward_normal(0.0f, 1.0f, 0.0f);

    const Vector3F outgoing = interaction(incident, outward_normal);

    EXPECT_LT(outgoing.dot(outward_normal), 0.0f);
}
