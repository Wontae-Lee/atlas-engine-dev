#include <atlas/material/neutron.h>

#include <atlas/material/material.h>
#include <atlas/material/material_type.h>
#include <atlas/material/molecule.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::Material;
using atlas::MaterialType;
using atlas::Molecule;
using atlas::Neutron;
using atlas::tol;

}

/**
 * Neutron mirrors Molecule's layout (eight floats), so it too must be trivially
 * copyable to be a valid Material leaf.
 */
static_assert(std::is_trivially_copyable_v<Neutron>,
              "Neutron must be trivially copyable for device buffers");

TEST(Neutron, DefaultConstructionMatchesDocumentedDefaults) {
    const Neutron leaf {};

    EXPECT_NEAR(leaf.mass(), 0.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 0.0f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 0.5f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.0f, tol);
}

TEST(Neutron, ExplicitConstructionExposesEveryProperty) {
    const Neutron leaf(2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 8.5f, 0.55f, 1.45f);

    EXPECT_NEAR(leaf.mass(), 2.5f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 3.5f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 4.5f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 5.5f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 6.5f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 8.5f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 0.55f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.45f, tol);
}

TEST(Neutron, NuclearStateIsNotModelledSoBehavesLikeMolecule) {
    /**
     * A neutron stores no nuclear data; it is eight plain floats like a
     * molecule, so identical inputs give identical getter outputs.
     */
    const Neutron neutron(2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 8.5f, 0.55f, 1.45f);
    const Molecule molecule(2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 8.5f, 0.55f, 1.45f);

    EXPECT_NEAR(neutron.mass(), molecule.mass(), tol);
    EXPECT_NEAR(neutron.translational_energy(), molecule.translational_energy(), tol);
    EXPECT_NEAR(neutron.reference_diameter(), molecule.reference_diameter(), tol);
    EXPECT_NEAR(neutron.scattering_parameter(), molecule.scattering_parameter(), tol);
}

TEST(Neutron, StoresNonPhysicalValuesWithoutValidation) {
    const Neutron leaf(-1.0f, 0.0f, 0.0f, 0.0f, -2.0f, -3.0f, 0.5f, 1.0f);

    EXPECT_NEAR(leaf.mass(), -1.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), -2.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), -3.0f, tol);
}

TEST(Neutron, RoundTripsThroughMaterialUmbrella) {
    const Neutron leaf(2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 8.5f, 0.55f, 1.45f);
    const Material material(leaf);

    EXPECT_EQ(material.type, MaterialType::neutron);
    EXPECT_NEAR(material.mass(), leaf.mass(), tol);
    EXPECT_NEAR(material.rotational_energy(), leaf.rotational_energy(), tol);
    EXPECT_NEAR(material.reference_temperature(), leaf.reference_temperature(), tol);
    EXPECT_NEAR(material.viscosity_index(), leaf.viscosity_index(), tol);
}
