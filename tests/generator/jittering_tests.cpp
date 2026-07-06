#include <atlas/generator/jittering_generator.h>

#include <atlas/generator/generate.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::GenerateType;
using atlas::JitteringGenerator;

}

TEST(JitteringGenerator, BuilderConstructsConfiguredGenerator) {
    const auto generator = JitteringGenerator::builder()
                               .with_base_value(2.0f)
                               .with_jitter_radius(0.5f)
                               .with_seed(11u)
                               .build();

    EXPECT_EQ(generator.type(), GenerateType::jittering);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::jittering);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::jittering);
    EXPECT_FLOAT_EQ(generator.param0(), 2.0f);
    EXPECT_FLOAT_EQ(generator.param1(), 0.5f);
}

TEST(JitteringGenerator, BuilderRejectsInvalidConfiguration) {
    EXPECT_THROW(
        static_cast<void>(JitteringGenerator::builder()
                              .with_base_value(1.0f)
                              .with_jitter_radius(-0.1f)
                              .build()),
        std::runtime_error);
}

TEST(JitteringGenerator, GenerateReturnsFiniteVectorNearBaseValue) {
    const auto generator = JitteringGenerator::builder()
                               .with_base_value(1.0f)
                               .with_jitter_radius(0.25f)
                               .with_seed(5u)
                               .build();

    const auto sample = generator.generate();

    EXPECT_TRUE(atlas::isfinite(sample));
    EXPECT_GE(sample.x, 0.75f);
    EXPECT_LE(sample.x, 1.25f);
    EXPECT_GE(sample.y, 0.75f);
    EXPECT_LE(sample.y, 1.25f);
    EXPECT_GE(sample.z, 0.75f);
    EXPECT_LE(sample.z, 1.25f);
}
