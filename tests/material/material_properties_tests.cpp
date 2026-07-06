#include <atlas/material/material_properties.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;

constexpr float tol = 1e-6f;

} // namespace

TEST(MaterialProperties, DefaultConstructionLeavesOptionalsEmpty) {
    // Arrange and act: default-construct material properties.
    const MaterialProperties properties;

    // Assert: defaults match the molecule baseline and optionals are empty.
    EXPECT_EQ(properties.type, MaterialType::molecule);
    EXPECT_NEAR(properties.mass, 0.0f, tol);
    EXPECT_NEAR(properties.molecular_mass, 0.0f, tol);
    EXPECT_FALSE(properties.translational_energy.has_value());
    EXPECT_FALSE(properties.charge.has_value());
}

TEST(MaterialProperties, BuilderConstructsRecordFromExplicitMass) {
    // Arrange and act: build a fully populated material record.
    const auto properties = MaterialProperties::builder()
                                .with_type(MaterialType::ion)
                                .with_mass(10.0f)
                                .with_molecular_mass(2.0f)
                                .with_translational_energy(1.0f)
                                .with_rotational_energy(2.0f)
                                .with_vibrational_energy(3.0f)
                                .with_rotational_dof(2)
                                .with_vibrational_dof(4)
                                .with_rotational_temperature(750.0f)
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
                                .with_rotational_relaxation_probability(0.25f)
                                .with_vibrational_relaxation_probability(0.5f)
                                .with_rotational_relaxation_coefficients(1.0f, 2.0f, 3.0f)
                                .with_vibrational_relaxation_coefficients(4.0f, 5.0f)
                                .with_rest_density(7.0f)
                                .with_pressure_coefficient(8.0f)
                                .with_dynamic_viscosity(9.0f)
                                .with_electronic_energy(11.0f)
                                .with_charge(2)
                                .build();

    // Assert: scalar and optional fields preserve the builder input.
    EXPECT_EQ(properties.type, MaterialType::ion);
    EXPECT_NEAR(properties.mass, 10.0f, tol);
    ASSERT_TRUE(properties.translational_energy.has_value());
    ASSERT_TRUE(properties.rotational_energy.has_value());
    ASSERT_TRUE(properties.vibrational_energy.has_value());
    ASSERT_TRUE(properties.rotational_dof.has_value());
    ASSERT_TRUE(properties.vibrational_dof.has_value());
    ASSERT_TRUE(properties.rotational_temperature.has_value());
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
    ASSERT_TRUE(properties.rotational_relaxation_probability.has_value());
    ASSERT_TRUE(properties.vibrational_relaxation_probability.has_value());
    ASSERT_TRUE(properties.rotational_relaxation_c1.has_value());
    ASSERT_TRUE(properties.rotational_relaxation_c2.has_value());
    ASSERT_TRUE(properties.rotational_relaxation_c3.has_value());
    ASSERT_TRUE(properties.vibrational_relaxation_c1.has_value());
    ASSERT_TRUE(properties.vibrational_relaxation_c2.has_value());
    ASSERT_TRUE(properties.rest_density.has_value());
    ASSERT_TRUE(properties.pressure_coefficient.has_value());
    ASSERT_TRUE(properties.dynamic_viscosity.has_value());
    ASSERT_TRUE(properties.electronic_energy.has_value());
    ASSERT_TRUE(properties.charge.has_value());
    EXPECT_NEAR(properties.molecular_mass, 2.0f, tol);
    EXPECT_NEAR(*properties.translational_energy, 1.0f, tol);
    EXPECT_NEAR(*properties.rotational_energy, 2.0f, tol);
    EXPECT_NEAR(*properties.vibrational_energy, 3.0f, tol);
    EXPECT_EQ(*properties.rotational_dof, 2);
    EXPECT_EQ(*properties.vibrational_dof, 4);
    EXPECT_NEAR(*properties.rotational_temperature, 750.0f, tol);
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
    EXPECT_NEAR(*properties.rotational_relaxation_probability, 0.25f, tol);
    EXPECT_NEAR(*properties.vibrational_relaxation_probability, 0.5f, tol);
    EXPECT_NEAR(*properties.rotational_relaxation_c1, 1.0f, tol);
    EXPECT_NEAR(*properties.rotational_relaxation_c2, 2.0f, tol);
    EXPECT_NEAR(*properties.rotational_relaxation_c3, 3.0f, tol);
    EXPECT_NEAR(*properties.vibrational_relaxation_c1, 4.0f, tol);
    EXPECT_NEAR(*properties.vibrational_relaxation_c2, 5.0f, tol);
    EXPECT_NEAR(*properties.rest_density, 7.0f, tol);
    EXPECT_NEAR(*properties.pressure_coefficient, 8.0f, tol);
    EXPECT_NEAR(*properties.dynamic_viscosity, 9.0f, tol);
    EXPECT_NEAR(*properties.electronic_energy, 11.0f, tol);
    EXPECT_EQ(*properties.charge, 2);
}

TEST(MaterialProperties, BuilderRequiresExplicitMass) {
    // Assert: molecular mass without explicit mass is rejected.
    EXPECT_THROW(
        MaterialProperties::builder()
            .with_molecular_mass(2.5f)
            .build(),
        std::invalid_argument);
}

TEST(MaterialProperties, MakeHostSharedReturnsUsableRecord) {
    // Act: build a shared material record.
    const auto properties = MaterialProperties::builder()
                                .with_mass(3.0f)
                                .with_molecular_mass(1.0f)
                                .make_host_shared();

    // Assert: the shared record exists and preserves input values.
    ASSERT_NE(properties, nullptr);
    EXPECT_NEAR(properties->mass, 3.0f, tol);
}

TEST(MaterialProperties, BuilderRejectsInvalidMassInputsImmediately) {
    // Assert: non-positive mass values are rejected immediately.
    EXPECT_THROW(
        MaterialProperties::builder()
            .with_mass(0.0f),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties::builder()
            .with_molecular_mass(0.0f),
        std::invalid_argument);
}

TEST(MaterialProperties, BuilderRejectsInvalidInternalEnergyInputsImmediately) {
    EXPECT_THROW(
        MaterialProperties::builder()
            .with_rotational_dof(1),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties::builder()
            .with_vibrational_dof(3),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties::builder()
            .with_rotational_temperature(0.0f),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties::builder()
            .with_characteristic_vibrational_temperature(0.0f),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties::builder()
            .with_rotational_relaxation_probability(1.5f),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties::builder()
            .with_vibrational_relaxation_probability(-0.1f),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties::builder()
            .with_rotational_relaxation_coefficients(0.0f, 1.0f, 1.0f),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties::builder()
            .with_vibrational_relaxation_coefficients(0.0f, 1.0f),
        std::invalid_argument);
}

TEST(MaterialProperties, BuilderRejectsMissingOrInconsistentMassConfiguration) {
    // Assert: exactly one mass field is not enough to build a valid material.
    EXPECT_THROW(
        MaterialProperties::builder()
            .with_mass(5.0f)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        MaterialProperties::builder()
            .with_molecular_mass(2.0f)
            .build(),
        std::invalid_argument);
}
