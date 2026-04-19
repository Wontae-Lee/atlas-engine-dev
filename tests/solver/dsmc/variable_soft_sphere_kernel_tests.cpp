#include "../../utilities/tests_utils.h"

#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

#include <testkit/testkit.h>

namespace {

using T = float;

atlas::MatrialProperties<T>
make_properties() {
    return atlas::MatrialProperties<T>::builder()
        .with_type(atlas::MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_collision_diameter(1.0f)
        .with_scattering_parameter(1.25f)
        .build();
}

} // namespace

TEST(VariableSoftSphereKernel, CrossSectionIsPositiveForPositiveSpeed) {
    const auto properties = make_properties();

    EXPECT_GT(atlas::system::VariableSoftSphereKernel<T>::cross_section(properties, properties, 2.0f), 0.0f);
}

TEST(VariableSoftSphereKernel, CollisionPreservesFiniteVelocities) {
    const auto properties = make_properties();
    atlas::Vector3<T> lhs(1, 0, 0);
    atlas::Vector3<T> rhs(-1, 0, 0);

    atlas::system::VariableSoftSphereKernel<T> {}(lhs, rhs, properties, properties);

    EXPECT_TRUE(atlas::test::is_finite_vec(lhs));
    EXPECT_TRUE(atlas::test::is_finite_vec(rhs));
}
