#include "../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/maxwell_sigma_generator.h>

#include <testkit/testkit.h>

namespace {

using T = float;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(MaxwellSigmaGenerator, OperatorReturnsZeroForNonPositiveSigma) {
    const atlas::fluid::MaxwellSigmaGenerateOperator<T> generator(13u);

    EXPECT_TRUE(atlas::test::vec_near(generator.generate(0.0f), atlas::Vector3<T>(0, 0, 0), 0.0f));
    EXPECT_TRUE(atlas::test::vec_near(generator.generate(-1.0f), atlas::Vector3<T>(0, 0, 0), 0.0f));
}

TEST(MaxwellSigmaGenerator, DirectConstructorExposesConfiguredParameters) {
    const atlas::fluid::MaxwellSigmaGenerator<T> generator(0.75f, 31u);

    EXPECT_EQ(generator.type(), atlas::fluid::GenerateType::maxwell_sigma);
    EXPECT_NEAR(generator.param0(), 0.75f, kEps);
    EXPECT_NEAR(generator.param1(), 1.0f, kEps);
    EXPECT_EQ(generator.generate_operator().type, atlas::fluid::GenerateType::maxwell_sigma);
    EXPECT_EQ(generator.make_generate_operator().type, atlas::fluid::GenerateType::maxwell_sigma);
    EXPECT_TRUE(atlas::test::is_finite_vec(generator.generate()));
}

TEST(MaxwellSigmaGenerator, BuilderConstructsConfiguredGenerator) {
    const auto generator = atlas::fluid::MaxwellSigmaGenerator<T>::builder()
                               .with_sigma(0.5f)
                               .with_seed(9u)
                               .build();

    EXPECT_NEAR(generator.param0(), 0.5f, kEps);
}

TEST(MaxwellSigmaGenerator, BuilderRejectsMissingOrInvalidSigma) {
    EXPECT_THROW(
        atlas::fluid::MaxwellSigmaGenerator<T>::builder()
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::MaxwellSigmaGenerator<T>::builder()
            .with_sigma(0.0f)
            .build(),
        std::runtime_error);
}

TEST(MaxwellSigmaGenerator, MakeHostSharedReturnsUsableGenerator) {
    const auto generator = atlas::fluid::MaxwellSigmaGenerator<T>::builder()
                               .with_sigma(0.5f)
                               .make_host_shared();

    ASSERT_NE(generator, nullptr);
    EXPECT_EQ(generator->type(), atlas::fluid::GenerateType::maxwell_sigma);
}
