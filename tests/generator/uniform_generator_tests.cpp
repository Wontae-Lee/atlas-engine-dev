#include "../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/uniform_generator.h>

#include <gtest/gtest.h>

namespace {

using T = float;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(UniformGenerator, OperatorProducesBoundedFiniteSamples) {
    const atlas::fluid::UniformGenerateOperator<T> generator(13u);
    const auto sample = generator.generate(-2.0f, 3.0f);

    EXPECT_TRUE(atlas::test::is_finite_vec(sample));
    EXPECT_GE(sample.x, -2.0f);
    EXPECT_LE(sample.x, 3.0f);
    EXPECT_GE(sample.y, -2.0f);
    EXPECT_LE(sample.y, 3.0f);
    EXPECT_GE(sample.z, -2.0f);
    EXPECT_LE(sample.z, 3.0f);
}

TEST(UniformGenerator, DirectConstructorExposesConfiguredParameters) {
    const atlas::fluid::UniformGenerator<T> generator(-1.5f, 2.5f, 31u);

    EXPECT_EQ(generator.type(), atlas::fluid::GenerateType::uniform);
    EXPECT_NEAR(generator.param0(), -1.5f, kEps);
    EXPECT_NEAR(generator.param1(), 2.5f, kEps);
    EXPECT_EQ(generator.generate_operator().type, atlas::fluid::GenerateType::uniform);
    EXPECT_EQ(generator.make_generate_operator().type, atlas::fluid::GenerateType::uniform);
    EXPECT_TRUE(atlas::test::is_finite_vec(generator.generate()));
}

TEST(UniformGenerator, BuilderConstructsConfiguredGenerator) {
    const auto generator = atlas::fluid::UniformGenerator<T>::builder()
                               .with_min_value(-4.0f)
                               .with_max_value(6.0f)
                               .with_seed(7u)
                               .build();

    EXPECT_NEAR(generator.param0(), -4.0f, kEps);
    EXPECT_NEAR(generator.param1(), 6.0f, kEps);
}

TEST(UniformGenerator, BuilderRejectsMissingOrInvalidBounds) {
    EXPECT_THROW(
        atlas::fluid::UniformGenerator<T>::builder()
            .with_max_value(1.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::UniformGenerator<T>::builder()
            .with_min_value(2.0f)
            .with_max_value(2.0f)
            .build(),
        std::runtime_error);
}

TEST(UniformGenerator, MakeHostSharedReturnsUsableGenerator) {
    const auto generator = atlas::fluid::UniformGenerator<T>::builder()
                               .with_min_value(-1.0f)
                               .with_max_value(1.0f)
                               .make_host_shared();

    ASSERT_NE(generator, nullptr);
    EXPECT_EQ(generator->type(), atlas::fluid::GenerateType::uniform);
}
