#include <atlas/material/solid.h>

#include <atlas/material/material.h>
#include <atlas/material/material_type.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::Material;
using atlas::MaterialType;
using atlas::Solid;
using atlas::tol;

}

/**
 * Solid stores a single float and must remain trivially copyable so the whole
 * Material union stays device-copyable.
 */
static_assert(std::is_trivially_copyable_v<Solid>,
              "Solid must be trivially copyable for device buffers");

/**
 * The mass constructor is explicit, so a bare float never implicitly converts
 * to a Solid even though it is constructible from one.
 */
static_assert(std::is_constructible_v<Solid, float>,
              "Solid must be constructible from a mass");
static_assert(!std::is_convertible_v<float, Solid>,
              "Solid's mass constructor must stay explicit");

TEST(Solid, DefaultConstructionHasZeroMassAndUnitStubs) {
    const Solid leaf {};

    EXPECT_NEAR(leaf.mass(), 0.0f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.rotational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.vibrational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 1.0f, tol);
    EXPECT_NEAR(leaf.reference_temperature(), 1.0f, tol);
    EXPECT_NEAR(leaf.viscosity_index(), 1.0f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.0f, tol);
}

TEST(Solid, MassConstructorStoresMassWhileStubsStayOne) {
    const Solid leaf(12.5f);

    EXPECT_NEAR(leaf.mass(), 12.5f, tol);
    EXPECT_NEAR(leaf.translational_energy(), 1.0f, tol);
    EXPECT_NEAR(leaf.reference_diameter(), 1.0f, tol);
    EXPECT_NEAR(leaf.scattering_parameter(), 1.0f, tol);
}

TEST(Solid, StubValuesAreIndependentOfMass) {
    /** The 1.0f stubs are constants; changing the mass never changes them. */
    const Solid light(0.001f);
    const Solid heavy(1000.0f);

    EXPECT_NEAR(light.reference_diameter(), heavy.reference_diameter(), tol);
    EXPECT_NEAR(light.viscosity_index(), heavy.viscosity_index(), tol);
    EXPECT_NEAR(light.reference_diameter(), 1.0f, tol);
    EXPECT_NEAR(heavy.reference_diameter(), 1.0f, tol);
}

TEST(Solid, RoundTripsThroughMaterialUmbrella) {
    const Solid leaf(12.5f);
    const Material material(leaf);

    EXPECT_EQ(material.type, MaterialType::solid);
    EXPECT_NEAR(material.mass(), leaf.mass(), tol);
    /** Forwarded non-mass getters surface the leaf's 1.0f stubs. */
    EXPECT_NEAR(material.reference_diameter(), 1.0f, tol);
    EXPECT_NEAR(material.translational_energy(), 1.0f, tol);
}
