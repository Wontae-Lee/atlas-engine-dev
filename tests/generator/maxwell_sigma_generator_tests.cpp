#include "../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/maxwell_sigma_generator.h>

#include <testkit/testkit.h>

namespace {

using atlas::GenerateType;
using atlas::MaxwellSigmaGenerateOperator;
using atlas::MaxwellSigmaGenerator;
using atlas::Vector3F;
using atlas::test::is_finite_vec;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(MaxwellSigmaGenerator, OperatorReturnsZeroForNonPositiveSigma) {
    // Arrange: create a deterministic Maxwell sigma generate operator.
    const MaxwellSigmaGenerateOperator<float> generator(13u);

    // Assert: non-positive sigma values produce a zero vector.
    EXPECT_TRUE(vec_near(generator.generate(0.0f), Vector3F(0, 0, 0), 0.0f));
    EXPECT_TRUE(vec_near(generator.generate(-1.0f), Vector3F(0, 0, 0), 0.0f));
}

TEST(MaxwellSigmaGenerator, DirectConstructorExposesConfiguredParameters) {
    // Arrange and act: construct a Maxwell sigma generator directly.
    const MaxwellSigmaGenerator<float> generator(0.75f, 31u);

    // Assert: configured parameters and operator type are exposed.
    EXPECT_EQ(generator.type(), GenerateType::maxwell_sigma);
    EXPECT_NEAR(generator.param0(), 0.75f, tol);
    EXPECT_NEAR(generator.param1(), 1.0f, tol);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::maxwell_sigma);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::maxwell_sigma);
    EXPECT_TRUE(is_finite_vec(generator.generate()));
}

TEST(MaxwellSigmaGenerator, BuilderConstructsConfiguredGenerator) {
    // Act: build a Maxwell sigma generator through the builder.
    const auto generator = MaxwellSigmaGenerator<float>::builder()
                               .with_sigma(0.5f)
                               .with_seed(9u)
                               .build();

    // Assert: builder values are preserved.
    EXPECT_NEAR(generator.param0(), 0.5f, tol);
}

TEST(MaxwellSigmaGenerator, BuilderRejectsMissingOrInvalidSigma) {
    // Assert: missing or invalid sigma values are rejected.
    EXPECT_THROW(
        MaxwellSigmaGenerator<float>::builder()
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        MaxwellSigmaGenerator<float>::builder()
            .with_sigma(0.0f)
            .build(),
        std::runtime_error);
}

TEST(MaxwellSigmaGenerator, MakeHostSharedReturnsUsableGenerator) {
    // Act: build a shared Maxwell sigma generator.
    const auto generator = MaxwellSigmaGenerator<float>::builder()
                               .with_sigma(0.5f)
                               .make_host_shared();

    // Assert: the shared generator exists and has the expected type.
    ASSERT_NE(generator, nullptr);
    EXPECT_EQ(generator->type(), GenerateType::maxwell_sigma);
}
