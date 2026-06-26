#include "../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/uniform_generator.h>

#include <testkit/testkit.h>

namespace {

using atlas::GenerateType;
using atlas::UniformGenerateOperator;
using atlas::UniformGenerator;
using atlas::test::is_finite_vec;
using atlas::tol;

} // namespace

TEST(UniformGenerator, OperatorProducesBoundedFiniteSamples) {
    // Arrange: create a deterministic uniform generate operator.
    const UniformGenerateOperator<float> generator(13u);

    // Act: generate a bounded sample.
    const auto sample = generator.generate(-2.0f, 3.0f);

    // Assert: every component is finite and inside the requested range.
    EXPECT_TRUE(is_finite_vec(sample));
    EXPECT_GE(sample.x, -2.0f);
    EXPECT_LE(sample.x, 3.0f);
    EXPECT_GE(sample.y, -2.0f);
    EXPECT_LE(sample.y, 3.0f);
    EXPECT_GE(sample.z, -2.0f);
    EXPECT_LE(sample.z, 3.0f);
}

TEST(UniformGenerator, DirectConstructorExposesConfiguredParameters) {
    // Arrange and act: construct a uniform generator directly.
    const UniformGenerator<float> generator(-1.5f, 2.5f, 31u);

    // Assert: configured bounds and operator type are exposed.
    EXPECT_EQ(generator.type(), GenerateType::uniform);
    EXPECT_NEAR(generator.param0(), -1.5f, tol);
    EXPECT_NEAR(generator.param1(), 2.5f, tol);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::uniform);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::uniform);
    EXPECT_TRUE(is_finite_vec(generator.generate()));
}

TEST(UniformGenerator, BuilderConstructsConfiguredGenerator) {
    // Act: build a uniform generator through the builder.
    const auto generator = UniformGenerator<float>::builder()
                               .with_min_value(-4.0f)
                               .with_max_value(6.0f)
                               .with_seed(7u)
                               .build();

    // Assert: builder values are preserved.
    EXPECT_NEAR(generator.param0(), -4.0f, tol);
    EXPECT_NEAR(generator.param1(), 6.0f, tol);
}

TEST(UniformGenerator, BuilderRejectsMissingOrInvalidBounds) {
    // Assert: missing or invalid bounds are rejected.
    EXPECT_THROW(
        UniformGenerator<float>::builder()
            .with_max_value(1.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        UniformGenerator<float>::builder()
            .with_min_value(2.0f)
            .with_max_value(2.0f)
            .build(),
        std::runtime_error);
}

TEST(UniformGenerator, MakeHostSharedReturnsUsableGenerator) {
    // Act: build a shared uniform generator.
    const auto generator = UniformGenerator<float>::builder()
                               .with_min_value(-1.0f)
                               .with_max_value(1.0f)
                               .make_host_shared();

    // Assert: the shared generator exists and has the expected type.
    ASSERT_NE(generator, nullptr);
    EXPECT_EQ(generator->type(), GenerateType::uniform);
}
