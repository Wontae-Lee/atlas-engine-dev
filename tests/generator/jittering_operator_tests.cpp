#include <atlas/generator/generate_operator.h>
#include <atlas/generator/jittering_operator.h>

#include <testkit/testkit.h>

namespace {

using T = float;

} // namespace

TEST(JitteringOperator, BuilderConstructsConfiguredGenerator) {
    const auto generator = atlas::fluid::JitteringOperator<T>::builder()
                               .with_base_value(2.0f)
                               .with_jitter_radius(0.5f)
                               .with_seed(11u)
                               .build();

    EXPECT_EQ(generator.type(), atlas::fluid::GenerateType::jittering);
    EXPECT_EQ(generator.generate_operator().type, atlas::fluid::GenerateType::jittering);
    EXPECT_EQ(generator.make_generate_operator().type, atlas::fluid::GenerateType::jittering);
    EXPECT_FLOAT_EQ(generator.param0(), 2.0f);
    EXPECT_FLOAT_EQ(generator.param1(), 0.5f);
}

TEST(JitteringOperator, BuilderRejectsInvalidConfiguration) {
    EXPECT_THROW(
        atlas::fluid::JitteringOperator<T>::builder()
            .with_base_value(1.0f)
            .with_jitter_radius(-0.1f)
            .build(),
        std::runtime_error);
}

TEST(JitteringOperator, GenerateReturnsFiniteVectorNearBaseValue) {
    const auto generator = atlas::fluid::JitteringOperator<T>::builder()
                               .with_base_value(1.0f)
                               .with_jitter_radius(0.25f)
                               .with_seed(5u)
                               .build();

    const auto sample = generator.generate();

    EXPECT_TRUE(std::isfinite(sample.x));
    EXPECT_TRUE(std::isfinite(sample.y));
    EXPECT_TRUE(std::isfinite(sample.z));
    EXPECT_GE(sample.x, 0.75f);
    EXPECT_LE(sample.x, 1.25f);
    EXPECT_GE(sample.y, 0.75f);
    EXPECT_LE(sample.y, 1.25f);
    EXPECT_GE(sample.z, 0.75f);
    EXPECT_LE(sample.z, 1.25f);
}
