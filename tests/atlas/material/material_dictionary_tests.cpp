#include <atlas/material/material_dictionary.h>

#include <atlas/buffer/host_buffer.h>
#include <atlas/material/atom.h>
#include <atlas/material/ion.h>
#include <atlas/material/material.h>
#include <atlas/material/material_type.h>
#include <atlas/material/molecule.h>
#include <atlas/material/neutron.h>
#include <atlas/material/solid.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace {

using atlas::Atom;
using atlas::HostBuffer;
using atlas::Ion;
using atlas::Material;
using atlas::MaterialDictionary;
using atlas::MaterialType;
using atlas::Molecule;
using atlas::Neutron;
using atlas::Solid;
using atlas::tol;

Material
make_molecule(const float mass) {
    return Material(Molecule(mass, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.5f, 1.0f));
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
                          .with_material(Material(Atom(4.0f, 1.0f, 2.0f, 3.0f, 5.0f, 6.0f, 0.5f, 1.0f)))
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
    EXPECT_NEAR(solid.translational_energy(), 1.0f, tol);
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

TEST(MaterialDictionary, DefaultConstructedIsEmpty) {
    const MaterialDictionary dictionary {};

    EXPECT_TRUE(dictionary.empty());
    EXPECT_EQ(dictionary.size(), std::size_t { 0 });
}

TEST(MaterialDictionary, RoundTripsIonThroughDeviceBuffer) {
    auto dictionary = MaterialDictionary::builder()
                          .with_material(Material(Ion(5.0f, 1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 0.75f, 1.25f)))
                          .build();

    const Material ion = dictionary.materials()[0];

    EXPECT_EQ(ion.type, MaterialType::ion);
    EXPECT_NEAR(ion.mass(), 5.0f, tol);
    EXPECT_NEAR(ion.reference_temperature(), 7.0f, tol);
    EXPECT_NEAR(ion.scattering_parameter(), 1.25f, tol);
}

TEST(MaterialDictionary, RoundTripsNeutronThroughDeviceBuffer) {
    auto dictionary = MaterialDictionary::builder()
                          .with_material(Material(Neutron(6.0f, 1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 0.75f, 1.25f)))
                          .build();

    const Material neutron = dictionary.materials()[0];

    EXPECT_EQ(neutron.type, MaterialType::neutron);
    EXPECT_NEAR(neutron.mass(), 6.0f, tol);
    EXPECT_NEAR(neutron.vibrational_energy(), 3.0f, tol);
}

TEST(MaterialDictionary, PreservesSpeciesOrderAcrossMixedAppends) {
    // A single append and a batch append must share one contiguous id space.
    const HostBuffer<Material> batch { make_molecule(2.0f), make_molecule(3.0f) };

    auto dictionary = MaterialDictionary::builder()
                          .with_material(make_molecule(1.0f))
                          .with_materials(batch)
                          .build();

    ASSERT_EQ(dictionary.size(), std::size_t { 3 });

    const Material first = dictionary.materials()[0];
    const Material second = dictionary.materials()[1];
    const Material third = dictionary.materials()[2];
    EXPECT_NEAR(first.mass(), 1.0f, tol);
    EXPECT_NEAR(second.mass(), 2.0f, tol);
    EXPECT_NEAR(third.mass(), 3.0f, tol);
}

TEST(MaterialDictionary, BuilderIsEmptyAfterBuild) {
    auto builder = MaterialDictionary::builder();
    builder.with_material(make_molecule(1.0f));

    const auto dictionary = builder.build();
    EXPECT_EQ(dictionary.size(), std::size_t { 1 });

    // build() clears its staging buffer, so the reused builder has nothing left.
    EXPECT_THROW(static_cast<void>(builder.build()), std::runtime_error);
}

TEST(MaterialDictionary, MoveConstructionTransfersOwnership) {
    auto source = MaterialDictionary::builder()
                      .with_material(make_molecule(2.0f))
                      .build();

    const MaterialDictionary moved = std::move(source);

    EXPECT_EQ(moved.size(), std::size_t { 1 });
    const Material material = moved.materials()[0];
    EXPECT_NEAR(material.mass(), 2.0f, tol);
    EXPECT_TRUE(source.empty());
}

TEST(MaterialDictionary, MoveAssignmentTransfersOwnership) {
    MaterialDictionary target {};

    target = MaterialDictionary::builder()
                 .with_material(make_molecule(3.0f))
                 .build();

    EXPECT_EQ(target.size(), std::size_t { 1 });
    const Material material = target.materials()[0];
    EXPECT_NEAR(material.mass(), 3.0f, tol);
}

TEST(MaterialDictionary, MutableMaterialsAccessorExposesTable) {
    auto dictionary = MaterialDictionary::builder()
                          .with_material(make_molecule(2.0f))
                          .build();

    auto& buffer = dictionary.materials();

    EXPECT_EQ(buffer.size(), std::size_t { 1 });
    const Material material = buffer[0];
    EXPECT_NEAR(material.mass(), 2.0f, tol);
}
