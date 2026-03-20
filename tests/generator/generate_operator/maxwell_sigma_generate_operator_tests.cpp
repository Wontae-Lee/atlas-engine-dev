#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(MaxwellSigmaGenerateOperator, GenerateProducesFiniteVelocities) {
    DeviceBuffer<Vector3<double>> values(64);

    MaxwellSigmaGenerateOperator<double> {}.generate(values, 2.5, 5u);

    EXPECT_TRUE(test::all_finite_points(values));
}

TEST(MaxwellSigmaGenerateOperator, GenerateWithSameSeedProducesSameSequence) {
    DeviceBuffer<Vector3<double>> lhs(32);
    DeviceBuffer<Vector3<double>> rhs(32);

    MaxwellSigmaGenerateOperator<double> {}.generate(lhs, 1.25, 13u);
    MaxwellSigmaGenerateOperator<double> {}.generate(rhs, 1.25, 13u);

    EXPECT_TRUE(test::point_buffers_near(lhs, rhs, eps));
}

TEST(MaxwellSigmaGenerateOperator, ZeroSigmaProducesZeroVelocities) {
    DeviceBuffer<Vector3<double>> values(16);

    MaxwellSigmaGenerateOperator<double> {}.generate(values, 0.0, 2u);

    for (const auto& value : values) {
        EXPECT_TRUE(test::vec_near(value, Vector3<double>(0.0, 0.0, 0.0), eps));
    }
}
