#include "../../utilities/test_utils.h"

#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

#include <testkit/testkit.h>

#include <cmath>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::Vector3F;
using atlas::VariableSoftSphereKernel;
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

TEST(VariableSoftSphereKernel, CollisionScatteringSamplesVssDistribution) {
    // Arrange: keep the incoming relative direction fixed and vary the center velocity seed.
    const auto properties = make_properties();
    constexpr int sample_count = 4096;
    constexpr float scattering_parameter = 1.25f;

    Vector3F mean(0.0f);
    Vector3F second_moment(0.0f);
    int forward_count = 0;

    // Act: recover the post-collision relative direction from many deterministic hash samples.
    for (int i = 0; i < sample_count; ++i) {
        const auto index = static_cast<float>(i);
        const Vector3F center(
            index * 0.61803399f + 0.17f,
            index * 1.41421356f + 0.31f,
            index * 2.71828183f + 0.53f);

        Vector3F lhs = center + Vector3F(1.0f, 0.0f, 0.0f);
        Vector3F rhs = center - Vector3F(1.0f, 0.0f, 0.0f);

        VariableSoftSphereKernel<float> {}(lhs, rhs, properties, properties);

        const Vector3F direction = (lhs - rhs).normalized();
        mean += direction;
        second_moment += direction * direction;
        forward_count += direction.x > 0.0f ? 1 : 0;
    }

    mean /= static_cast<float>(sample_count);
    second_moment /= static_cast<float>(sample_count);

    const float expected_axis_mean = (scattering_parameter - 1.0f) / (scattering_parameter + 1.0f);
    const float expected_axis_second = 4.0f * scattering_parameter / (scattering_parameter + 2.0f)
        - 4.0f * scattering_parameter / (scattering_parameter + 1.0f)
        + 1.0f;
    const float expected_tangent_second = (1.0f - expected_axis_second) * 0.5f;
    const float expected_forward_ratio = 1.0f - std::pow(0.5f, scattering_parameter);

    // Assert: VSS follows cos(chi)=2*u^(1/alpha)-1 with uniform azimuth.
    EXPECT_NEAR(mean.x, expected_axis_mean, 0.04f);
    EXPECT_NEAR(mean.y, 0.0f, 0.035f);
    EXPECT_NEAR(mean.z, 0.0f, 0.035f);
    EXPECT_NEAR(second_moment.x, expected_axis_second, 0.04f);
    EXPECT_NEAR(second_moment.y, expected_tangent_second, 0.04f);
    EXPECT_NEAR(second_moment.z, expected_tangent_second, 0.04f);
    EXPECT_NEAR(static_cast<float>(forward_count) / static_cast<float>(sample_count), expected_forward_ratio, 0.04f);
}
