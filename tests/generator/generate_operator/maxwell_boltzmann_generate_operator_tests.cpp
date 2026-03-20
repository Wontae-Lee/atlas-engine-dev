#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cmath>

using namespace atlas;

TEST(MaxwellBoltzmannGenerateOperator, InvalidParametersClearTheBuffer) {
    DeviceBuffer<Vector3<double>> values(8, Vector3<double>(1.0, 2.0, 3.0));

    MaxwellBoltzmannGenerateOperator<double> {}.generate(
        values,
        0.0,
        1.0,
        Vector3<double>(0.0, 0.0, 0.0),
        1u);

    EXPECT_TRUE(values.empty());
}

TEST(MaxwellBoltzmannGenerateOperator, GenerateMatchesSigmaOperatorPlusBulkVelocityForSameSeed) {
    constexpr double temperature    = 300.0;
    constexpr double molecular_mass = 4.65e-26;
    const Vector3<double> bulk_velocity(2.0, -1.0, 0.5);
    const double sigma = std::sqrt(static_cast<double>(boltzmann_constant) * temperature / molecular_mass);

    DeviceBuffer<Vector3<double>> expected(24);
    DeviceBuffer<Vector3<double>> actual(24);

    MaxwellSigmaGenerateOperator<double> {}.generate(expected, sigma, 17u);
    for (auto& value : expected) {
        value += bulk_velocity;
    }

    MaxwellBoltzmannGenerateOperator<double> {}.generate(
        actual,
        temperature,
        molecular_mass,
        bulk_velocity,
        17u);

    EXPECT_TRUE(test::point_buffers_near(expected, actual, eps));
}

TEST(MaxwellBoltzmannGenerateOperator, GenerateProducesFiniteVelocities) {
    DeviceBuffer<Vector3<double>> values(64);

    MaxwellBoltzmannGenerateOperator<double> {}.generate(
        values,
        300.0,
        4.65e-26,
        Vector3<double>(0.0, 0.0, 0.0),
        9u);

    EXPECT_TRUE(test::all_finite_points(values));
}
