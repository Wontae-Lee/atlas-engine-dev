#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cmath>

using namespace atlas;

TEST(MaxwellBoltzmannGenerateOperator, InvalidParametersReturnZeroVelocity) {
    MaxwellBoltzmannGenerateOperator<double> op(1u);
    const auto value = op.generate(0.0, 1.0);

    EXPECT_TRUE(test::vec_near(value, Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(MaxwellBoltzmannGenerateOperator, GenerateMatchesSigmaOperatorPlusBulkVelocityForSameSeed) {
    constexpr double temperature    = 300.0;
    constexpr double molecular_mass = 4.65e-26;
    const Vector3<double> bulk_velocity(2.0, -1.0, 0.5);
    const double sigma = std::sqrt(static_cast<double>(boltzmann_constant) * temperature / molecular_mass);
    MaxwellSigmaGenerateOperator<double> sigma_op(17u);
    MaxwellBoltzmannGenerateOperator<double> mb_op(17u, bulk_velocity);
    const auto expected = sigma_op.generate(sigma) + bulk_velocity;
    const auto actual   = mb_op.generate(temperature, molecular_mass);

    EXPECT_TRUE(test::vec_near(expected, actual, eps));
}

TEST(MaxwellBoltzmannGenerateOperator, GenerateProducesFiniteVelocities) {
    MaxwellBoltzmannGenerateOperator<double> op(9u);
    const auto value = op.generate(300.0, 4.65e-26);

    EXPECT_TRUE(test::is_finite_vec(value));
}
