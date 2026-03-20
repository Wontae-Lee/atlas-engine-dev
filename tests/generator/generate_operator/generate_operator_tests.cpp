#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(GenerateOperator, UniformDispatchMatchesConcreteOperator) {
    DeviceBuffer<Vector3<double>> expected(20);
    DeviceBuffer<Vector3<double>> actual(20);

    UniformGenerateOperator<double> {}.generate(expected, -1.0, 2.0, 5u);
    GenerateOperator<double>(GenerateType::uniform).generate(actual, -1.0, 2.0, 5u);

    EXPECT_TRUE(test::point_buffers_near(expected, actual, eps));
}

TEST(GenerateOperator, MaxwellSigmaDispatchMatchesConcreteOperator) {
    DeviceBuffer<Vector3<double>> expected(20);
    DeviceBuffer<Vector3<double>> actual(20);

    MaxwellSigmaGenerateOperator<double> {}.generate(expected, 1.75, 6u);
    GenerateOperator<double>(GenerateType::maxwell_sigma).generate(actual, 1.75, 0.0, 6u);

    EXPECT_TRUE(test::point_buffers_near(expected, actual, eps));
}

TEST(GenerateOperator, MaxwellBoltzmannDispatchMatchesConcreteOperator) {
    DeviceBuffer<Vector3<double>> expected(20);
    DeviceBuffer<Vector3<double>> actual(20);
    const Vector3<double> bulk_velocity(1.0, 2.0, 3.0);

    MaxwellBoltzmannGenerateOperator<double> {}.generate(
        expected,
        350.0,
        4.65e-26,
        bulk_velocity,
        10u);

    GenerateOperator<double>(GenerateType::maxwell_boltzmann).generate(
        actual,
        350.0,
        4.65e-26,
        bulk_velocity,
        10u);

    EXPECT_TRUE(test::point_buffers_near(expected, actual, eps));
}
