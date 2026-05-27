#include "../utilities/test_utils.h"

#include <atlas/material/material_properties.h>

#include <testkit/testkit.h>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::tol;

} // namespace

TEST(MatrialProperties, DefaultConstructionLeavesOptionalsEmpty) {
    // Arrange and act: default-construct material properties.
    const MaterialProperties<float> properties;

    // Assert: defaults match the molecule baseline and optionals are empty.
    EXPECT_EQ(properties.type, MaterialType::Molecule);
    EXPECT_NEAR(properties.mass, 0.0f, tol);
    EXPECT_NEAR(properties.molecular_mass, 0.0f, tol);
    EXPECT_FALSE(properties.translational_energy.has_value());
    EXPECT_FALSE(properties.charge.has_value());
}

TEST(MatrialProperties, BuilderConstructsRecordFromExplicitMass) {
    // Arrange and act: build a fully populated material record.
    const auto properties = MaterialProperties<float>::builder()
                                .with_type(MaterialType::Ion)
                                .with_mass(10.0f)
                                .with_molecular_mass(2.0f)
                                .with_translational_energy(1.0f)
                                .with_rotational_energy(2.0f)
                                .with_vibrational_energy(3.0f)
                                .with_characteristic_vibrational_temperature(1000.0f)
                                .with_max_vibrational_quantum(12)
                                .with_gamma_quant(0.5f)
                                .with_interaction_id(20)
                                .with_fully_ionized(false)
                                .with_polyatomic_molecule(true)
                                .with_species_id(7)
                                .with_reference_diameter(4.0f)
                                .with_reference_temperature(5.0f)
                                .with_viscosity_index(0.75f)
                                .with_scattering_parameter(6.0f)
                                .with_rest_density(7.0f)
                                .with_pressure_coefficient(8.0f)
                                .with_dynamic_viscosity(9.0f)
                                .with_smoothing_length(10.0f)
                                .with_electronic_energy(11.0f)
                                .with_charge(2)
                                .build();

    // Assert: scalar and optional fields preserve the builder input.
    EXPECT_EQ(properties.type, MaterialType::Ion);
    EXPECT_NEAR(properties.mass, 10.0f, tol);
    ASSERT_TRUE(properties.translational_energy.has_value());
    ASSERT_TRUE(properties.rotational_energy.has_value());
    ASSERT_TRUE(properties.vibrational_energy.has_value());
    ASSERT_TRUE(properties.characteristic_vibrational_temperature.has_value());
    ASSERT_TRUE(properties.max_vibrational_quantum.has_value());
    ASSERT_TRUE(properties.gamma_quant.has_value());
    ASSERT_TRUE(properties.interaction_id.has_value());
    ASSERT_TRUE(properties.fully_ionized.has_value());
    ASSERT_TRUE(properties.polyatomic_molecule.has_value());
    ASSERT_TRUE(properties.species_id.has_value());
    ASSERT_TRUE(properties.reference_diameter.has_value());
    ASSERT_TRUE(properties.reference_temperature.has_value());
    ASSERT_TRUE(properties.viscosity_index.has_value());
    ASSERT_TRUE(properties.scattering_parameter.has_value());
    ASSERT_TRUE(properties.rest_density.has_value());
    ASSERT_TRUE(properties.pressure_coefficient.has_value());
    ASSERT_TRUE(properties.dynamic_viscosity.has_value());
    ASSERT_TRUE(properties.smoothing_length.has_value());
    ASSERT_TRUE(properties.electronic_energy.has_value());
    ASSERT_TRUE(properties.charge.has_value());
    EXPECT_NEAR(properties.molecular_mass, 2.0f, tol);
    EXPECT_NEAR(*properties.translational_energy, 1.0f, tol);
    EXPECT_NEAR(*properties.rotational_energy, 2.0f, tol);
    EXPECT_NEAR(*properties.vibrational_energy, 3.0f, tol);
    EXPECT_NEAR(*properties.characteristic_vibrational_temperature, 1000.0f, tol);
    EXPECT_EQ(*properties.max_vibrational_quantum, 12);
    EXPECT_NEAR(*properties.gamma_quant, 0.5f, tol);
    EXPECT_EQ(*properties.interaction_id, 20);
    EXPECT_FALSE(*properties.fully_ionized);
    EXPECT_TRUE(*properties.polyatomic_molecule);
    EXPECT_EQ(*properties.species_id, 7);
    EXPECT_NEAR(*properties.reference_diameter, 4.0f, tol);
    EXPECT_NEAR(*properties.reference_temperature, 5.0f, tol);
    EXPECT_NEAR(*properties.viscosity_index, 0.75f, tol);
    EXPECT_NEAR(*properties.scattering_parameter, 6.0f, tol);
    EXPECT_NEAR(*properties.rest_density, 7.0f, tol);
    EXPECT_NEAR(*properties.pressure_coefficient, 8.0f, tol);
    EXPECT_NEAR(*properties.dynamic_viscosity, 9.0f, tol);
    EXPECT_NEAR(*properties.smoothing_length, 10.0f, tol);
    EXPECT_NEAR(*properties.electronic_energy, 11.0f, tol);
    EXPECT_EQ(*properties.charge, 2);
}

TEST(MatrialProperties, BuilderRequiresExplicitMass) {
    // Assert: molecular mass without explicit mass is rejected.
    EXPECT_THROW(
        MaterialProperties<float>::builder()
            .with_molecular_mass(2.5f)
            .build(),
        std::invalid_argument);
}

TEST(MatrialProperties, MakeHostSharedReturnsUsableRecord) {
    // Act: build a shared material record.
    const auto properties = MaterialProperties<float>::builder()
                                .with_mass(3.0f)
                                .with_molecular_mass(1.0f)
                                .make_host_shared();

    // Assert: the shared record exists and preserves input values.
    ASSERT_NE(properties, nullptr);
    EXPECT_NEAR(properties->mass, 3.0f, tol);
}

TEST(MatrialProperties, BuilderRejectsInvalidMassInputsImmediately) {
    // Assert: non-positive mass values are rejected immediately.
    EXPECT_THROW(
        MaterialProperties<float>::builder()
            .with_mass(0.0f),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties<float>::builder()
            .with_molecular_mass(0.0f),
        std::invalid_argument);
}

TEST(MatrialProperties, BuilderRejectsMissingOrInconsistentMassConfiguration) {
    // Assert: exactly one mass field is not enough to build a valid material.
    EXPECT_THROW(
        MaterialProperties<float>::builder()
            .with_mass(5.0f)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties<float>::builder()
            .with_molecular_mass(2.0f)
            .build(),
        std::invalid_argument);
}
