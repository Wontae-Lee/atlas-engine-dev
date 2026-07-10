#include <atlas/material/ion.h>

#include <atlas/material/material.h>
#include <atlas/material/material_type.h>
#include <atlas/material/molecule.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::Ion;
using atlas::Material;
using atlas::MaterialType;
using atlas::Molecule;
using atlas::tol;

}

/**
 * Ion holds nothing but floats, so it stays trivially copyable and can be
 * stored in a DeviceBuffer<Material> like every other leaf.
 */
static_assert(std::is_trivially_copyable_v<Ion>,
              "Ion must be trivially copyable for device buffers");

TEST(Ion, DefaultConstructionMatchesDocumentedDefaults) {
    const Ion leaf {};

    EXPECT_NEAR(leaf.mass(), 0.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 0.0f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 0.5f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.0f, tol);
}

TEST(Ion, ExplicitConstructionExposesEveryProperty) {
    const Ion leaf(2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 0.6f, 1.4f);

    EXPECT_NEAR(leaf.mass(), 2.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 3.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 4.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 5.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 6.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 8.0f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 0.6f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.4f, tol);
}

TEST(Ion, ChargeIsNotModelledSoBehavesLikeMolecule) {
    /**
     * The charge itself is deliberately not stored (see ion.h / material.md):
     * an Ion built from the same inputs as a Molecule is observationally
     * identical across all eight getters.
     */
    const Ion ion(2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 0.6f, 1.4f);
    const Molecule molecule(2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 0.6f, 1.4f);

    EXPECT_NEAR(ion.mass(), molecule.mass(), tol);
    EXPECT_NEAR(ion.reference_diameter(), molecule.reference_diameter(), tol);
    EXPECT_NEAR(ion.reference_temperature(), molecule.reference_temperature(), tol);
    EXPECT_NEAR(ion.viscosity_index(), molecule.viscosity_index(), tol);
    EXPECT_NEAR(ion.scattering_parameter(), molecule.scattering_parameter(), tol);
}

TEST(Ion, StoresNonPhysicalValuesWithoutValidation) {
    const Ion leaf(-1.0f, 0.0f, 0.0f, 0.0f, -2.0f, -3.0f, 0.5f, 1.0f);

    EXPECT_NEAR(leaf.mass(), -1.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), -2.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), -3.0f, tol);
}

TEST(Ion, RoundTripsThroughMaterialUmbrella) {
    const Ion leaf(2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 0.6f, 1.4f);
    const Material material(leaf);

    EXPECT_EQ(material.type, MaterialType::ion);
    EXPECT_NEAR(material.mass(), leaf.mass(), tol);
    EXPECT_NEAR(material.vibrational_energy(), leaf.vibrational_energy(), tol);
    EXPECT_NEAR(material.reference_temperature(), leaf.reference_temperature(), tol);
    EXPECT_NEAR(material.scattering_parameter(), leaf.scattering_parameter(), tol);
}
