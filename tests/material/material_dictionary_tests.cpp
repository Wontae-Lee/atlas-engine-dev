#include <atlas/material/material_dictionary.h>

#include <atlas/buffer/host_buffer.h>
#include <atlas/material/atom.h>
#include <atlas/material/material.h>
#include <atlas/material/material_type.h>
#include <atlas/material/molecule.h>
#include <atlas/material/solid.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace {

using atlas::Atom;
using atlas::HostBuffer;
using atlas::Material;
using atlas::MaterialDictionary;
using atlas::MaterialType;
using atlas::Molecule;
using atlas::Solid;
using atlas::tol;

Material
make_molecule(const float mass) {
    return Material(Molecule(mass, 1.0f, 2.0f, 3.0f));
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
                          .with_material(Material(Atom(4.0f, 1.0f, 2.0f, 3.0f)))
                          .build();

    EXPECT_EQ(dictionary.size(), std::size_t { 2 });
    EXPECT_FALSE(dictionary.empty());

    const Material first = dictionary.materials()[0];
    EXPECT_EQ(first.type, MaterialType::molecule);
    EXPECT_NEAR(first.mass(), 2.0f, tol);
    EXPECT_NEAR(first.rotational_energy(), 2.0f, tol);

    const Material second = dictionary.materials()[1];
    EXPECT_EQ(second.type, MaterialType::atom);
    EXPECT_NEAR(second.mass(), 4.0f, tol);
}

TEST(MaterialDictionary, RoundTripsSolidThroughDeviceBuffer) {
    auto dictionary = MaterialDictionary::builder()
                          .with_material(Material(Solid(7.0f)))
                          .build();

    const Material solid = dictionary.materials()[0];

    EXPECT_EQ(solid.type, MaterialType::solid);
    EXPECT_NEAR(solid.mass(), 7.0f, tol);
    EXPECT_THROW(static_cast<void>(solid.translational_energy()), std::runtime_error);
}

TEST(MaterialDictionary, WithMaterialsAppendsRange) {
    const HostBuffer<Material> materials { make_molecule(1.0f), make_molecule(2.0f), make_molecule(3.0f) };

    auto dictionary = MaterialDictionary::builder()
                          .with_materials(materials)
                          .build();

    EXPECT_EQ(dictionary.size(), std::size_t { 3 });

    const Material third = dictionary.materials()[2];
    EXPECT_NEAR(third.mass(), 3.0f, tol);
}

TEST(MaterialDictionary, MakeHostSharedBuildsDictionary) {
    const auto dictionary = MaterialDictionary::builder()
                                .with_material(make_molecule(2.0f))
                                .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(dictionary));
    EXPECT_EQ(dictionary->size(), std::size_t { 1 });
}
