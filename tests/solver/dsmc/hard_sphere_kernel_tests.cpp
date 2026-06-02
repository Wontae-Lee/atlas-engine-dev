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
        .with_reference_diameter(1.0f)
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

TEST(HardSphereKernel, CollisionScatteringSamplesUniformSphere) {
    // Arrange: keep the incoming relative direction fixed and vary the center velocity seed.
    const auto properties = make_properties();
    constexpr int sample_count = 4096;

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

        HardSphereKernel<float> {}(lhs, rhs, properties, properties);

        const Vector3F direction = (lhs - rhs).normalized();
        mean += direction;
        second_moment += direction * direction;
        forward_count += direction.x > 0.0f ? 1 : 0;
    }

    mean /= static_cast<float>(sample_count);
    second_moment /= static_cast<float>(sample_count);

    // Assert: an isotropic unit-sphere distribution has zero mean and E[x^2]=E[y^2]=E[z^2]=1/3.
    EXPECT_NEAR(mean.x, 0.0f, 0.035f);
    EXPECT_NEAR(mean.y, 0.0f, 0.035f);
    EXPECT_NEAR(mean.z, 0.0f, 0.035f);
    EXPECT_NEAR(second_moment.x, 1.0f / 3.0f, 0.04f);
    EXPECT_NEAR(second_moment.y, 1.0f / 3.0f, 0.04f);
    EXPECT_NEAR(second_moment.z, 1.0f / 3.0f, 0.04f);
    EXPECT_NEAR(static_cast<float>(forward_count) / static_cast<float>(sample_count), 0.5f, 0.035f);
}
