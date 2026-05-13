#include "../../utilities/test_utils.h"

#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::Vector3F;
using atlas::system::VariableHardSphereKernel;
using atlas::test::is_finite_vec;

MaterialProperties<float>
make_properties() {
    return MaterialProperties<float>::builder()
        .with_type(MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_collision_diameter(1.0f)
        .with_viscosity_index(0.75f)
        .build();
}

} // namespace

TEST(VariableHardSphereKernel, CrossSectionIsPositiveForPositiveSpeed) {
    // Arrange: create representative material properties.
    const auto properties = make_properties();

    // Assert: variable hard-sphere cross section is positive for positive speed.
    EXPECT_GT(VariableHardSphereKernel<float>::cross_section(properties, properties, 2.0f), 0.0f);
}

TEST(VariableHardSphereKernel, CollisionPreservesFiniteVelocities) {
    // Arrange: create a pair of finite velocities and material properties.
    const auto properties = make_properties();
    Vector3F lhs(1, 0, 0);
    Vector3F rhs(-1, 0, 0);

    // Act: apply a variable hard-sphere collision.
    VariableHardSphereKernel<float> {}(lhs, rhs, properties, properties);

    // Assert: collision output remains finite.
    EXPECT_TRUE(is_finite_vec(lhs));
    EXPECT_TRUE(is_finite_vec(rhs));
}
