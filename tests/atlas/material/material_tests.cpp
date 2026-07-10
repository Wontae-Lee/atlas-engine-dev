#include <atlas/material/material.h>

#include <atlas/material/atom.h>
#include <atlas/material/ion.h>
#include <atlas/material/material_type.h>
#include <atlas/material/molecule.h>
#include <atlas/material/neutron.h>
#include <atlas/material/solid.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>

namespace {

using atlas::Atom;
using atlas::Ion;
using atlas::Material;
using atlas::MaterialType;
using atlas::Molecule;
using atlas::Neutron;
using atlas::Solid;
using atlas::tol;

}

static_assert(std::is_trivially_copyable_v<Material>,
              "Material must be trivially copyable for device buffers");
static_assert(std::is_trivially_copyable_v<Molecule>);
static_assert(std::is_trivially_copyable_v<Atom>);
static_assert(std::is_trivially_copyable_v<Ion>);
static_assert(std::is_trivially_copyable_v<Neutron>);
static_assert(std::is_trivially_copyable_v<Solid>);

TEST(Molecule, GettersReturnConstructedValues) {
    const Molecule leaf(2.0f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 0.75f, 1.25f);

    EXPECT_NEAR(leaf.mass(), 2.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 1.5f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 2.5f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 3.5f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 4.5f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 5.5f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 0.75f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.25f, tol);
}

TEST(Molecule, DefaultConstructsZeroed) {
    const Molecule leaf {};

    EXPECT_NEAR(leaf.mass(), 0.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 0.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 0.0f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 0.5f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.0f, tol);
}

TEST(Atom, GettersReturnConstructedValues) {
    const Atom leaf(4.0f, 1.0f, 2.0f, 3.0f, 5.0f, 6.0f, 0.75f, 1.25f);

    EXPECT_NEAR(leaf.mass(), 4.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 2.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 3.0f, tol);
}

TEST(Ion, GettersReturnConstructedValues) {
    const Ion leaf(5.0f, 1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 0.75f, 1.25f);

    EXPECT_NEAR(leaf.mass(), 5.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 2.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 3.0f, tol);
}

TEST(Neutron, GettersReturnConstructedValues) {
    const Neutron leaf(6.0f, 1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 0.75f, 1.25f);

    EXPECT_NEAR(leaf.mass(), 6.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 2.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 3.0f, tol);
}

TEST(Solid, CarriesOnlyMass) {
    const Solid leaf(7.0f);

    EXPECT_NEAR(leaf.mass(), 7.0f, tol);
    EXPECT_EQ(sizeof(Solid), sizeof(float));
}

TEST(Solid, AbsentPropertiesReadAsOne) {
    const Solid leaf(7.0f);

    EXPECT_NEAR(leaf.translational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 1.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 1.0f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 1.0f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.0f, tol);
}

TEST(Material, DefaultConstructsMolecule) {
    const Material material {};

    EXPECT_EQ(material.type, MaterialType::molecule);
    EXPECT_NEAR(material.mass(), 0.0f, tol);
}

TEST(Material, WrapsMoleculeLeafAndExposesGetters) {
    const Material material(Molecule(2.0f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 0.75f, 1.25f));

    EXPECT_EQ(material.type, MaterialType::molecule);
    EXPECT_NEAR(material.mass(), 2.0f, tol);
    EXPECT_NEAR(material.translational_energy(), 1.5f, tol);
    EXPECT_NEAR(material.rotational_energy(), 2.5f, tol);
    EXPECT_NEAR(material.vibrational_energy(), 3.5f, tol);
    EXPECT_NEAR(material.reference_diameter(), 4.5f, tol);
    EXPECT_NEAR(material.reference_temperature(), 5.5f, tol);
    EXPECT_NEAR(material.viscosity_index(), 0.75f, tol);
    EXPECT_NEAR(material.scattering_parameter(), 1.25f, tol);
}

TEST(Material, WrapsAtomLeaf) {
    const Material material(Atom(4.0f, 1.0f, 2.0f, 3.0f, 5.0f, 6.0f, 0.75f, 1.25f));

    EXPECT_EQ(material.type, MaterialType::atom);
    EXPECT_NEAR(material.mass(), 4.0f, tol);
    EXPECT_NEAR(material.vibrational_energy(), 3.0f, tol);
}

TEST(Material, WrapsIonLeaf) {
    const Material material(Ion(5.0f, 1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 0.75f, 1.25f));

    EXPECT_EQ(material.type, MaterialType::ion);
    EXPECT_NEAR(material.mass(), 5.0f, tol);
    EXPECT_NEAR(material.translational_energy(), 1.0f, tol);
}

TEST(Material, WrapsNeutronLeaf) {
    const Material material(Neutron(6.0f, 1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 0.75f, 1.25f));

    EXPECT_EQ(material.type, MaterialType::neutron);
    EXPECT_NEAR(material.mass(), 6.0f, tol);
    EXPECT_NEAR(material.rotational_energy(), 2.0f, tol);
}

TEST(Material, WrapsSolidLeafAndExposesMass) {
    const Material material(Solid(7.0f));

    EXPECT_EQ(material.type, MaterialType::solid);
    EXPECT_NEAR(material.mass(), 7.0f, tol);
}

TEST(Material, SolidAbsentPropertiesReadAsOne) {
    const Material material(Solid(7.0f));

    EXPECT_NEAR(material.translational_energy(), 1.0f, tol);
    EXPECT_NEAR(material.rotational_energy(), 1.0f, tol);
    EXPECT_NEAR(material.vibrational_energy(), 1.0f, tol);
    EXPECT_NEAR(material.reference_diameter(), 1.0f, tol);
    EXPECT_NEAR(material.reference_temperature(), 1.0f, tol);
    EXPECT_NEAR(material.viscosity_index(), 1.0f, tol);
    EXPECT_NEAR(material.scattering_parameter(), 1.0f, tol);
}

TEST(Material, CopyPreservesActiveLeaf) {
    const Material material(Ion(5.0f, 1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 0.75f, 1.25f));
    const Material copy = material;

    EXPECT_EQ(copy.type, MaterialType::ion);
    EXPECT_NEAR(copy.mass(), 5.0f, tol);
    EXPECT_NEAR(copy.vibrational_energy(), 3.0f, tol);
}

TEST(Material, AssignmentReplacesActiveLeaf) {
    Material material(Molecule(2.0f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 0.75f, 1.25f));

    material = Material(Solid(7.0f));

    EXPECT_EQ(material.type, MaterialType::solid);
    EXPECT_NEAR(material.mass(), 7.0f, tol);
    EXPECT_NEAR(material.rotational_energy(), 1.0f, tol);
}
