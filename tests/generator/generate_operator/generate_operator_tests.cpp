#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(GenerateOperator, UniformDispatchMatchesConcreteOperator) {
    const auto expected = UniformGenerateOperator<double>(5u).generate(-1.0, 2.0);
    const auto actual   = GenerateOperator<double>(GenerateType::uniform, 5u).generate(-1.0, 2.0);

    EXPECT_TRUE(test::vec_near(expected, actual, eps));
}

TEST(GenerateOperator, MaxwellSigmaDispatchMatchesConcreteOperator) {
    const auto expected = MaxwellSigmaGenerateOperator<double>(6u).generate(1.75);
    const auto actual   = GenerateOperator<double>(GenerateType::maxwell_sigma, 6u).generate(1.75, 0.0);

    EXPECT_TRUE(test::vec_near(expected, actual, eps));
}

TEST(GenerateOperator, MaxwellBoltzmannDispatchMatchesConcreteOperator) {
    const Vector3<double> bulk_velocity(1.0, 2.0, 3.0);
    const auto expected = MaxwellBoltzmannGenerateOperator<double>(10u, bulk_velocity).generate(
        350.0,
        4.65e-26);

    const auto actual = GenerateOperator<double>(
                            MaxwellBoltzmannGenerateOperator<double>(10u, bulk_velocity))
                            .generate(350.0, 4.65e-26);

    EXPECT_TRUE(test::vec_near(expected, actual, eps));
}
