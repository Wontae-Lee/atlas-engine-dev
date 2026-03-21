#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <stdexcept>

using namespace atlas;

TEST(MaxwellSigmaGenerator, GenerateUsesStoredSigmaAndSeed) {
    const auto expected = MaxwellSigmaGenerateOperator<double>(12u).generate(2.25);
    MaxwellSigmaGenerator<double> generator(2.25, 12u);
    const auto actual = generator.generate();

    EXPECT_TRUE(test::vec_near(expected, actual, eps));
}

TEST(MaxwellSigmaGenerator, TypeReturnsMaxwellSigma) {
    MaxwellSigmaGenerator<double> generator(1.0, 1u);

    EXPECT_EQ(generator.type(), GenerateType::maxwell_sigma);
}

TEST(MaxwellSigmaGenerator, BuilderBuildsAndRejectsNonPositiveSigma) {
    const auto generator = MaxwellSigmaGenerator<double>::builder()
                               .with_sigma(1.5)
                               .with_seed(9u)
                               .build();

    EXPECT_EQ(generator.type(), GenerateType::maxwell_sigma);

    EXPECT_THROW(
        MaxwellSigmaGenerator<double>::builder()
            .with_sigma(0.0)
            .build(),
        std::runtime_error);
}
