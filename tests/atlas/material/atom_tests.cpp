#include <atlas/material/atom.h>

#include <atlas/material/material.h>
#include <atlas/material/material_type.h>
#include <atlas/material/molecule.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::Atom;
using atlas::Material;
using atlas::MaterialType;
using atlas::Molecule;
using atlas::tol;

}

/**
 * Atom is a bundle of plain floats, so it must be trivially copyable to live in
 * a DeviceBuffer and be captured by value into a device lambda (see material.h).
 */
static_assert(std::is_trivially_copyable_v<Atom>,
              "Atom must be trivially copyable for device buffers");

TEST(Atom, DefaultConstructionMatchesDocumentedDefaults) {
    const Atom leaf {};

    EXPECT_NEAR(leaf.mass(), 0.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 0.0f, tol);
    /** The two VHS/VSS exponents keep their in-class defaults, not 0. */
    EXPECT_NEAR(leaf.viscosity_index(), 0.5f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.0f, tol);
}

TEST(Atom, ExplicitConstructionExposesEveryProperty) {
    const Atom leaf(1.25f, 2.5f, 3.75f, 5.0f, 6.25f, 7.5f, 0.65f, 1.35f);

    EXPECT_NEAR(leaf.mass(), 1.25f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 2.5f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 3.75f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 5.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 6.25f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 7.5f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 0.65f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.35f, tol);
}

TEST(Atom, RotationalAndVibrationalFieldsArePresentButNotForcedToZero) {
    /**
     * A monatomic species physically carries no rotational or vibrational
     * modes, yet the fields are still stored and echoed verbatim; the header
     * documents them as "nominally 0" but performs no clamping.
     */
    const Atom leaf(1.0f, 0.0f, 4.0f, 5.0f, 1.0f, 1.0f, 0.5f, 1.0f);

    EXPECT_NEAR(leaf.rotational_energy(), 4.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 5.0f, tol);
}

TEST(Atom, StoresNonPhysicalValuesWithoutValidation) {
    /**
     * There is no is_valid() and no clamping: degenerate inputs (non-positive
     * mass/diameter/temperature) are stored and returned unchanged.
     */
    const Atom leaf(-1.0f, 0.0f, 0.0f, 0.0f, -2.0f, -3.0f, 0.5f, 1.0f);

    EXPECT_NEAR(leaf.mass(), -1.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), -2.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), -3.0f, tol);
}

TEST(Atom, RoundTripsThroughMaterialUmbrella) {
    const Atom leaf(1.25f, 2.5f, 3.75f, 5.0f, 6.25f, 7.5f, 0.65f, 1.35f);
    const Material material(leaf);

    EXPECT_EQ(material.type, MaterialType::atom);
    EXPECT_NEAR(material.mass(), leaf.mass(), tol);
    EXPECT_NEAR(material.translational_energy(), leaf.translational_energy(), tol);
    EXPECT_NEAR(material.reference_diameter(), leaf.reference_diameter(), tol);
    EXPECT_NEAR(material.scattering_parameter(), leaf.scattering_parameter(), tol);
}
