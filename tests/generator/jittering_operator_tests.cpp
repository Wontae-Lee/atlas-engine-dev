#include "../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/jittering_operator.h>

#include <testkit/testkit.h>

namespace {

using atlas::GenerateType;
using atlas::JitteringOperator;
using atlas::test::is_finite_vec;

} // namespace

TEST(JitteringOperator, BuilderConstructsConfiguredGenerator) {
    // Act: build a configured jittering generator.
    const auto generator = JitteringOperator<float>::builder()
                               .with_base_value(2.0f)
                               .with_jitter_radius(0.5f)
                               .with_seed(11u)
                               .build();

    // Assert: parameters and operator type are exposed.
    EXPECT_EQ(generator.type(), GenerateType::jittering);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::jittering);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::jittering);
    EXPECT_FLOAT_EQ(generator.param0(), 2.0f);
    EXPECT_FLOAT_EQ(generator.param1(), 0.5f);
}

TEST(JitteringOperator, BuilderRejectsInvalidConfiguration) {
    // Assert: negative jitter radius is rejected.
    EXPECT_THROW(
        JitteringOperator<float>::builder()
            .with_base_value(1.0f)
            .with_jitter_radius(-0.1f)
            .build(),
        std::runtime_error);
}

TEST(JitteringOperator, GenerateReturnsFiniteVectorNearBaseValue) {
    // Arrange: build a jittering generator around a base value.
    const auto generator = JitteringOperator<float>::builder()
                               .with_base_value(1.0f)
                               .with_jitter_radius(0.25f)
                               .with_seed(5u)
                               .build();

    // Act: generate one jittered sample.
    const auto sample = generator.generate();

    // Assert: each component is finite and inside the configured jitter range.
    EXPECT_TRUE(is_finite_vec(sample));
    EXPECT_GE(sample.x, 0.75f);
    EXPECT_LE(sample.x, 1.25f);
    EXPECT_GE(sample.y, 0.75f);
    EXPECT_LE(sample.y, 1.25f);
    EXPECT_GE(sample.z, 0.75f);
    EXPECT_LE(sample.z, 1.25f);
}
