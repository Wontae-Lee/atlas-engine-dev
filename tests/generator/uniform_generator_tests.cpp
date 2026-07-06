#include <atlas/generator/uniform_generator.h>

#include <atlas/generator/generate.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::GenerateType;
using atlas::UniformGenerate;
using atlas::UniformGenerator;

}

TEST(UniformGenerator, OperatorProducesBoundedFiniteSamples) {
    const UniformGenerate generator(13u);

    const auto sample = generator.generate(-2.0f, 3.0f);

    EXPECT_TRUE(atlas::isfinite(sample));
    EXPECT_GE(sample.x, -2.0f);
    EXPECT_LE(sample.x, 3.0f);
    EXPECT_GE(sample.y, -2.0f);
    EXPECT_LE(sample.y, 3.0f);
    EXPECT_GE(sample.z, -2.0f);
    EXPECT_LE(sample.z, 3.0f);
}

TEST(UniformGenerator, DirectConstructorExposesConfiguredParameters) {
    const UniformGenerator generator(-1.5f, 2.5f, 31u);

    EXPECT_EQ(generator.type(), GenerateType::uniform);
    EXPECT_NEAR(generator.param0(), -1.5f, atlas::tol);
    EXPECT_NEAR(generator.param1(), 2.5f, atlas::tol);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::uniform);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::uniform);
    EXPECT_TRUE(atlas::isfinite(generator.generate()));
}

TEST(UniformGenerator, BuilderConstructsConfiguredGenerator) {
    const auto generator = UniformGenerator::builder()
                               .with_min_value(-4.0f)
                               .with_max_value(6.0f)
                               .with_seed(7u)
                               .build();

    EXPECT_NEAR(generator.param0(), -4.0f, atlas::tol);
    EXPECT_NEAR(generator.param1(), 6.0f, atlas::tol);
}

TEST(UniformGenerator, BuilderRejectsMissingOrInvalidBounds) {
    EXPECT_THROW(
        static_cast<void>(UniformGenerator::builder()
                              .with_max_value(1.0f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(UniformGenerator::builder()
                              .with_min_value(2.0f)
                              .with_max_value(2.0f)
                              .build()),
        std::runtime_error);
}

TEST(UniformGenerator, MakeHostSharedReturnsUsableGenerator) {
    const auto generator = UniformGenerator::builder()
                               .with_min_value(-1.0f)
                               .with_max_value(1.0f)
                               .make_host_shared();

    ASSERT_NE(generator, nullptr);
    EXPECT_EQ(generator->type(), GenerateType::uniform);
}
