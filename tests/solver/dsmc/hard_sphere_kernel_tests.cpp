#include "../../utilities/test_utils.h"

#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/hard_sphere_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::Vector3F;
using atlas::system::HardSphereKernel;
using atlas::test::is_finite_vec;

MaterialProperties<float>
make_properties() {
    return MaterialProperties<float>::builder()
        .with_type(MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_collision_diameter(1.0f)
        .build();
}

} // namespace

TEST(HardSphereKernel, CrossSectionIsPositiveWhenDiameterExists) {
    // Arrange: create representative material properties.
    const auto properties = make_properties();

    // Assert: hard-sphere cross section is positive for positive diameter.
    EXPECT_GT(HardSphereKernel<float>::cross_section(properties, properties), 0.0f);
}

TEST(HardSphereKernel, CollisionPreservesFiniteVelocities) {
    // Arrange: create a pair of finite velocities and material properties.
    const auto properties = make_properties();
    Vector3F lhs(1, 0, 0);
    Vector3F rhs(-1, 0, 0);

    // Act: apply a hard-sphere collision.
    HardSphereKernel<float> {}(lhs, rhs, properties, properties);

    // Assert: collision output remains finite.
    EXPECT_TRUE(is_finite_vec(lhs));
    EXPECT_TRUE(is_finite_vec(rhs));
}
