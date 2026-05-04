#include "../../utilities/tests_utils.h"

#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>

#include <testkit/testkit.h>

namespace {

using T = float;

atlas::MaterialProperties<T>
make_properties() {
    return atlas::MaterialProperties<T>::builder()
        .with_type(atlas::MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_collision_diameter(1.0f)
        .with_viscosity_index(0.75f)
        .build();
}

} // namespace

TEST(VariableHardSphereKernel, CrossSectionIsPositiveForPositiveSpeed) {
    const auto properties = make_properties();

    EXPECT_GT(atlas::system::VariableHardSphereKernel<T>::cross_section(properties, properties, 2.0f), 0.0f);
}

TEST(VariableHardSphereKernel, CollisionPreservesFiniteVelocities) {
    const auto properties = make_properties();
    atlas::Vector3<T> lhs(1, 0, 0);
    atlas::Vector3<T> rhs(-1, 0, 0);

    atlas::system::VariableHardSphereKernel<T> {}(lhs, rhs, properties, properties);

    EXPECT_TRUE(atlas::test::is_finite_vec(lhs));
    EXPECT_TRUE(atlas::test::is_finite_vec(rhs));
}
