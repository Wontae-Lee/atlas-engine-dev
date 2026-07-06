#include <atlas/generator/maxwell_sigma_generator.h>

#include <atlas/generator/generate.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::GenerateType;
using atlas::MaxwellSigmaGenerate;
using atlas::MaxwellSigmaGenerator;
using atlas::Float3;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, atlas::tol);
    EXPECT_NEAR(actual.y, expected.y, atlas::tol);
    EXPECT_NEAR(actual.z, expected.z, atlas::tol);
}

}

TEST(MaxwellSigmaGenerator, OperatorReturnsZeroForNonPositiveSigma) {
    const MaxwellSigmaGenerate generator(13u);

    expect_vec_near(generator.generate(0.0f), Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(generator.generate(-1.0f), Float3(0.0f, 0.0f, 0.0f));
}

TEST(MaxwellSigmaGenerator, DirectConstructorExposesConfiguredParameters) {
    const MaxwellSigmaGenerator generator(0.75f, 31u);

    EXPECT_EQ(generator.type(), GenerateType::maxwell_sigma);
    EXPECT_NEAR(generator.param0(), 0.75f, atlas::tol);
    EXPECT_NEAR(generator.param1(), 1.0f, atlas::tol);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::maxwell_sigma);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::maxwell_sigma);
    EXPECT_TRUE(atlas::isfinite(generator.generate()));
}

TEST(MaxwellSigmaGenerator, BuilderConstructsConfiguredGenerator) {
    const auto generator = MaxwellSigmaGenerator::builder()
                               .with_sigma(0.5f)
                               .with_seed(9u)
                               .build();

    EXPECT_NEAR(generator.param0(), 0.5f, atlas::tol);
}

TEST(MaxwellSigmaGenerator, BuilderRejectsMissingOrInvalidSigma) {
    EXPECT_THROW(
        static_cast<void>(MaxwellSigmaGenerator::builder()
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(MaxwellSigmaGenerator::builder()
                              .with_sigma(0.0f)
                              .build()),
        std::runtime_error);
}

TEST(MaxwellSigmaGenerator, MakeHostSharedReturnsUsableGenerator) {
    const auto generator = MaxwellSigmaGenerator::builder()
                               .with_sigma(0.5f)
                               .make_host_shared();

    ASSERT_NE(generator, nullptr);
    EXPECT_EQ(generator->type(), GenerateType::maxwell_sigma);
}
