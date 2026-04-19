#include "../utilities/tests_utils.h"

#include <atlas/material/material_properties.h>

#include <testkit/testkit.h>

namespace {

using T = float;

constexpr T kEps = static_cast<T>(1e-6);

} // namespace

TEST(MatrialProperties, DefaultConstructionLeavesOptionalsEmpty) {
    const atlas::system::MaterialProperties<T> properties;

    EXPECT_EQ(properties.type, atlas::system::MaterialType::Molecule);
    EXPECT_NEAR(properties.mass, 0.0f, kEps);
    EXPECT_NEAR(properties.molecular_mass, 0.0f, kEps);
    EXPECT_FALSE(properties.translational_energy.has_value());
    EXPECT_FALSE(properties.charge.has_value());
}

TEST(MatrialProperties, BuilderConstructsRecordFromExplicitMass) {
    const auto properties = atlas::system::MaterialProperties<T>::builder()
                                .with_type(atlas::system::MaterialType::Ion)
                                .with_mass(10.0f)
                                .with_molecular_mass(2.0f)
                                .with_translational_energy(1.0f)
                                .with_rotational_energy(2.0f)
                                .with_vibrational_energy(3.0f)
                                .with_species_id(7)
                                .with_collision_diameter(4.0f)
                                .with_viscosity_index(5.0f)
                                .with_scattering_parameter(6.0f)
                                .with_rest_density(7.0f)
                                .with_pressure_coefficient(8.0f)
                                .with_dynamic_viscosity(9.0f)
                                .with_smoothing_length(10.0f)
                                .with_electronic_energy(11.0f)
                                .with_charge(2)
                                .build();

    EXPECT_EQ(properties.type, atlas::system::MaterialType::Ion);
    EXPECT_NEAR(properties.mass, 10.0f, kEps);
    ASSERT_TRUE(properties.translational_energy.has_value());
    ASSERT_TRUE(properties.rotational_energy.has_value());
    ASSERT_TRUE(properties.vibrational_energy.has_value());
    ASSERT_TRUE(properties.species_id.has_value());
    ASSERT_TRUE(properties.collision_diameter.has_value());
    ASSERT_TRUE(properties.viscosity_index.has_value());
    ASSERT_TRUE(properties.scattering_parameter.has_value());
    ASSERT_TRUE(properties.rest_density.has_value());
    ASSERT_TRUE(properties.pressure_coefficient.has_value());
    ASSERT_TRUE(properties.dynamic_viscosity.has_value());
    ASSERT_TRUE(properties.smoothing_length.has_value());
    ASSERT_TRUE(properties.electronic_energy.has_value());
    ASSERT_TRUE(properties.charge.has_value());
    EXPECT_NEAR(properties.molecular_mass, 2.0f, kEps);
    EXPECT_NEAR(*properties.translational_energy, 1.0f, kEps);
    EXPECT_NEAR(*properties.rotational_energy, 2.0f, kEps);
    EXPECT_NEAR(*properties.vibrational_energy, 3.0f, kEps);
    EXPECT_EQ(*properties.species_id, 7);
    EXPECT_NEAR(*properties.collision_diameter, 4.0f, kEps);
    EXPECT_NEAR(*properties.viscosity_index, 5.0f, kEps);
    EXPECT_NEAR(*properties.scattering_parameter, 6.0f, kEps);
    EXPECT_NEAR(*properties.rest_density, 7.0f, kEps);
    EXPECT_NEAR(*properties.pressure_coefficient, 8.0f, kEps);
    EXPECT_NEAR(*properties.dynamic_viscosity, 9.0f, kEps);
    EXPECT_NEAR(*properties.smoothing_length, 10.0f, kEps);
    EXPECT_NEAR(*properties.electronic_energy, 11.0f, kEps);
    EXPECT_EQ(*properties.charge, 2);
}

TEST(MatrialProperties, BuilderRequiresExplicitMass) {
    EXPECT_THROW(
        atlas::system::MaterialProperties<T>::builder()
            .with_molecular_mass(2.5f)
            .build(),
        std::invalid_argument);
}

TEST(MatrialProperties, MakeHostSharedReturnsUsableRecord) {
    const auto properties = atlas::system::MaterialProperties<T>::builder()
                                .with_mass(3.0f)
                                .with_molecular_mass(1.0f)
                                .make_host_shared();

    ASSERT_NE(properties, nullptr);
    EXPECT_NEAR(properties->mass, 3.0f, kEps);
}

TEST(MatrialProperties, BuilderRejectsInvalidMassInputsImmediately) {
    EXPECT_THROW(
        atlas::system::MaterialProperties<T>::builder()
            .with_mass(0.0f),
        std::invalid_argument);

    EXPECT_THROW(
        atlas::system::MaterialProperties<T>::builder()
            .with_molecular_mass(0.0f),
        std::invalid_argument);
}

TEST(MatrialProperties, BuilderRejectsMissingOrInconsistentMassConfiguration) {
    EXPECT_THROW(
        atlas::system::MaterialProperties<T>::builder()
            .with_mass(5.0f)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        atlas::system::MaterialProperties<T>::builder()
            .with_molecular_mass(2.0f)
            .build(),
        std::invalid_argument);
}
