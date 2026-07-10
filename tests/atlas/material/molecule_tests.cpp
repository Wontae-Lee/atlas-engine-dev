#include <atlas/material/molecule.h>

#include <atlas/material/material.h>
#include <atlas/material/material_type.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::Material;
using atlas::MaterialType;
using atlas::Molecule;
using atlas::tol;

}

/**
 * Molecule is the default-active leaf of the Material union and, being a pure
 * float bundle, must be trivially copyable for the DeviceVariant machinery.
 */
static_assert(std::is_trivially_copyable_v<Molecule>,
              "Molecule must be trivially copyable for device buffers");

TEST(Molecule, DefaultConstructionMatchesDocumentedDefaults) {
    const Molecule leaf {};

    EXPECT_NEAR(leaf.mass(), 0.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 0.0f, tol);
    /** A default-constructed molecule is a hard sphere, not a degenerate one. */
    EXPECT_NEAR(leaf.viscosity_index(), 0.5f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.0f, tol);
}

TEST(Molecule, ExplicitConstructionExposesEveryProperty) {
    const Molecule leaf(3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 9.0f, 0.7f, 1.2f);

    EXPECT_NEAR(leaf.mass(), 3.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 4.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 5.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 6.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 7.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 9.0f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 0.7f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.2f, tol);
}

TEST(Molecule, StoresNonPhysicalValuesWithoutValidation) {
    /** No is_valid() exists; degenerate inputs are echoed back unchanged. */
    const Molecule leaf(-1.0f, 0.0f, 0.0f, 0.0f, -2.0f, -3.0f, 0.5f, 1.0f);

    EXPECT_NEAR(leaf.mass(), -1.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), -2.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), -3.0f, tol);
}

TEST(Molecule, RoundTripsThroughMaterialUmbrella) {
    const Molecule leaf(3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 9.0f, 0.7f, 1.2f);
    const Material material(leaf);

    EXPECT_EQ(material.type, MaterialType::molecule);
    EXPECT_NEAR(material.mass(), leaf.mass(), tol);
    EXPECT_NEAR(material.rotational_energy(), leaf.rotational_energy(), tol);
    EXPECT_NEAR(material.reference_diameter(), leaf.reference_diameter(), tol);
    EXPECT_NEAR(material.viscosity_index(), leaf.viscosity_index(), tol);
}
