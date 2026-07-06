#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

#include <atlas/material/material_properties.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::VariableSoftSphereKernel;
using atlas::Float3;

bool
is_finite_vec(const Float3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

MaterialProperties
make_properties() {
    return MaterialProperties::builder()
        .with_type(MaterialType::molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_reference_diameter(1.0f)
        .with_reference_temperature(1.0f)
        .with_viscosity_index(0.75f)
        .with_scattering_parameter(1.25f)
        .build();
}

}

TEST(VariableSoftSphereKernel, CrossSectionIsPositiveForPositiveSpeed) {
    const auto properties = make_properties();

    EXPECT_GT(VariableSoftSphereKernel::cross_section(properties, properties, 2.0f), 0.0f);
}

TEST(VariableSoftSphereKernel, CrossSectionUsesVhsTemperatureScaling) {
    const auto properties = make_properties();

    const float slow = VariableSoftSphereKernel::cross_section(properties, properties, 1.0f);
    const float fast = VariableSoftSphereKernel::cross_section(properties, properties, 4.0f);

    EXPECT_GT(slow, fast);
}

TEST(VariableSoftSphereKernel, CollisionPreservesFiniteVelocities) {
    const auto properties = make_properties();
    Float3 lhs(1.0f, 0.0f, 0.0f);
    Float3 rhs(-1.0f, 0.0f, 0.0f);

    VariableSoftSphereKernel {}(lhs, rhs, properties, properties);

    EXPECT_TRUE(is_finite_vec(lhs));
    EXPECT_TRUE(is_finite_vec(rhs));
}

TEST(VariableSoftSphereKernel, CollisionScatteringSamplesVssDistribution) {
    const auto properties = make_properties();
    constexpr int sample_count = 4096;
    constexpr float scattering_parameter = 1.25f;

    Float3 mean(0.0f, 0.0f, 0.0f);
    Float3 second_moment(0.0f, 0.0f, 0.0f);
    int forward_count = 0;

    for (int i = 0; i < sample_count; ++i) {
        const auto index = static_cast<float>(i);
        const Float3 center(
            index * 0.61803399f + 0.17f,
            index * 1.41421356f + 0.31f,
            index * 2.71828183f + 0.53f);

        Float3 lhs = center + Float3(1.0f, 0.0f, 0.0f);
        Float3 rhs = center - Float3(1.0f, 0.0f, 0.0f);

        VariableSoftSphereKernel {}(lhs, rhs, properties, properties);

        const Float3 direction = (lhs - rhs).normalized();
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

    EXPECT_NEAR(mean.x, expected_axis_mean, 0.04f);
    EXPECT_NEAR(mean.y, 0.0f, 0.035f);
    EXPECT_NEAR(mean.z, 0.0f, 0.035f);
    EXPECT_NEAR(second_moment.x, expected_axis_second, 0.04f);
    EXPECT_NEAR(second_moment.y, expected_tangent_second, 0.04f);
    EXPECT_NEAR(second_moment.z, expected_tangent_second, 0.04f);
    EXPECT_NEAR(static_cast<float>(forward_count) / static_cast<float>(sample_count), expected_forward_ratio, 0.04f);
}
