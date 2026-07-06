#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>

#include <atlas/material/material_properties.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::VariableHardSphereKernel;
using atlas::Vector3;

bool
is_finite_vec(const Vector3& v) {
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
        .build();
}

}

TEST(VariableHardSphereKernel, CrossSectionIsPositiveForPositiveSpeed) {
    const auto properties = make_properties();

    EXPECT_GT(VariableHardSphereKernel::cross_section(properties, properties, 2.0f), 0.0f);
}

TEST(VariableHardSphereKernel, CrossSectionDecreasesWithSpeedAboveHardSphereIndex) {
    const auto properties = make_properties();

    const float slow = VariableHardSphereKernel::cross_section(properties, properties, 1.0f);
    const float fast = VariableHardSphereKernel::cross_section(properties, properties, 4.0f);

    EXPECT_GT(slow, fast);
}

TEST(VariableHardSphereKernel, CollisionPreservesFiniteVelocities) {
    const auto properties = make_properties();
    Vector3 lhs(1.0f, 0.0f, 0.0f);
    Vector3 rhs(-1.0f, 0.0f, 0.0f);

    VariableHardSphereKernel {}(lhs, rhs, properties, properties);

    EXPECT_TRUE(is_finite_vec(lhs));
    EXPECT_TRUE(is_finite_vec(rhs));
}

TEST(VariableHardSphereKernel, CollisionScatteringSamplesUniformSphere) {
    const auto properties = make_properties();
    constexpr int sample_count = 4096;

    Vector3 mean(0.0f, 0.0f, 0.0f);
    Vector3 second_moment(0.0f, 0.0f, 0.0f);
    int forward_count = 0;

    for (int i = 0; i < sample_count; ++i) {
        const auto index = static_cast<float>(i);
        const Vector3 center(
            index * 0.61803399f + 0.17f,
            index * 1.41421356f + 0.31f,
            index * 2.71828183f + 0.53f);

        Vector3 lhs = center + Vector3(1.0f, 0.0f, 0.0f);
        Vector3 rhs = center - Vector3(1.0f, 0.0f, 0.0f);

        VariableHardSphereKernel {}(lhs, rhs, properties, properties);

        const Vector3 direction = (lhs - rhs).normalized();
        mean += direction;
        second_moment += direction * direction;
        forward_count += direction.x > 0.0f ? 1 : 0;
    }

    mean /= static_cast<float>(sample_count);
    second_moment /= static_cast<float>(sample_count);

    EXPECT_NEAR(mean.x, 0.0f, 0.035f);
    EXPECT_NEAR(mean.y, 0.0f, 0.035f);
    EXPECT_NEAR(mean.z, 0.0f, 0.035f);
    EXPECT_NEAR(second_moment.x, 1.0f / 3.0f, 0.04f);
    EXPECT_NEAR(second_moment.y, 1.0f / 3.0f, 0.04f);
    EXPECT_NEAR(second_moment.z, 1.0f / 3.0f, 0.04f);
    EXPECT_NEAR(static_cast<float>(forward_count) / static_cast<float>(sample_count), 0.5f, 0.035f);
}
