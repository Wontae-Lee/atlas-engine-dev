#include "../../utilities/test_utils.h"

#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::Vector3F;
using atlas::system::VariableSoftSphereKernel;
using atlas::test::is_finite_vec;

MaterialProperties<float>
make_properties() {
    return MaterialProperties<float>::builder()
        .with_type(MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_reference_diameter(1.0f)
        .with_reference_temperature(1.0f)
        .with_viscosity_index(0.75f)
        .with_scattering_parameter(1.25f)
        .build();
}

} // namespace

TEST(VariableSoftSphereKernel, CrossSectionIsPositiveForPositiveSpeed) {
    // Arrange: create representative material properties.
    const auto properties = make_properties();

    // Assert: variable soft-sphere cross section is positive for positive speed.
    EXPECT_GT(VariableSoftSphereKernel<float>::cross_section(properties, properties, 2.0f), 0.0f);
}

TEST(VariableSoftSphereKernel, CrossSectionUsesVhsTemperatureScaling) {
    // Arrange: create representative material properties.
    const auto properties = make_properties();

    // Act: evaluate the VSS total cross section at two relative speeds.
    const float slow = VariableSoftSphereKernel<float>::cross_section(properties, properties, 1.0f);
    const float fast = VariableSoftSphereKernel<float>::cross_section(properties, properties, 4.0f);

    // Assert: VSS uses the same VHS total cross-section law.
    EXPECT_GT(slow, fast);
}

TEST(VariableSoftSphereKernel, CollisionPreservesFiniteVelocities) {
    // Arrange: create a pair of finite velocities and material properties.
    const auto properties = make_properties();
    Vector3F lhs(1, 0, 0);
    Vector3F rhs(-1, 0, 0);

    // Act: apply a variable soft-sphere collision.
    VariableSoftSphereKernel<float> {}(lhs, rhs, properties, properties);

    // Assert: collision output remains finite.
    EXPECT_TRUE(is_finite_vec(lhs));
    EXPECT_TRUE(is_finite_vec(rhs));
}
