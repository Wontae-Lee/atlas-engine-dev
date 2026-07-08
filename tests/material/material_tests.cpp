#include <atlas/material/material.h>

#include <atlas/material/material_type.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::AtomMaterial;
using atlas::Material;
using atlas::MaterialType;
using atlas::MoleculeMaterial;
using atlas::tol;

}

static_assert(std::is_trivially_copyable_v<Material>,
              "Material must be trivially copyable for device buffers");

TEST(Material, DefaultConstructsMolecule) {
    const Material material {};

    EXPECT_EQ(material.type, MaterialType::molecule);
}

TEST(Material, WrapsLeafAndExposesFields) {
    MoleculeMaterial leaf;
    leaf.mass              = 2.0f;
    leaf.rotational_energy = 2.5f;

    const Material material(leaf);

    EXPECT_EQ(material.type, MaterialType::molecule);
    EXPECT_NEAR(material.mass(), 2.0f, tol);

    ASSERT_TRUE(material.rotational_energy().has_value());
    EXPECT_NEAR(*material.rotational_energy(), 2.5f, tol);

    EXPECT_FALSE(material.translational_energy().has_value());
    EXPECT_FALSE(material.vibrational_energy().has_value());
}

TEST(Material, WrapsAtomLeaf) {
    AtomMaterial leaf;
    leaf.mass = 5.0f;

    const Material material(leaf);

    EXPECT_EQ(material.type, MaterialType::atom);
    EXPECT_NEAR(material.mass(), 5.0f, tol);
}

TEST(Material, CopyPreservesActiveLeaf) {
    MoleculeMaterial leaf;
    leaf.mass = 3.0f;

    const Material material(leaf);
    const Material copy = material;

    EXPECT_EQ(copy.type, MaterialType::molecule);
    EXPECT_NEAR(copy.mass(), 3.0f, tol);
}
