#include <atlas/material/material_dictionary.h>

#include <atlas/material/material.h>
#include <atlas/material/material_type.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using atlas::AtomMaterial;
using atlas::Material;
using atlas::MaterialDictionary;
using atlas::MaterialType;
using atlas::MoleculeMaterial;
using atlas::tol;

Material
make_molecule(const float mass) {
    MoleculeMaterial leaf;
    leaf.mass = mass;
    return Material(leaf);
}

}

TEST(MaterialDictionary, BuilderRejectsEmpty) {
    EXPECT_THROW(
        static_cast<void>(MaterialDictionary::builder().build()),
        std::runtime_error);
}

TEST(MaterialDictionary, BuildsDeviceBufferOfMaterials) {
    auto dictionary = MaterialDictionary::builder()
                          .with_material(make_molecule(2.0f))
                          .with_material(Material(AtomMaterial {}))
                          .build();

    EXPECT_EQ(dictionary.size(), std::size_t { 2 });
    EXPECT_FALSE(dictionary.empty());

    const Material first = dictionary.materials()[0];
    EXPECT_EQ(first.type, MaterialType::molecule);
    EXPECT_NEAR(first.mass(), 2.0f, tol);

    const Material second = dictionary.materials()[1];
    EXPECT_EQ(second.type, MaterialType::atom);
}

TEST(MaterialDictionary, WithMaterialsAppendsRange) {
    const std::vector<Material> materials { make_molecule(1.0f), make_molecule(2.0f), make_molecule(3.0f) };

    auto dictionary = MaterialDictionary::builder()
                          .with_materials(materials)
                          .build();

    EXPECT_EQ(dictionary.size(), std::size_t { 3 });
}
