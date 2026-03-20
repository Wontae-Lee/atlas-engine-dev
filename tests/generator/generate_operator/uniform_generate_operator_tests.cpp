#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(UniformGenerateOperator, GenerateProducesValuesInsideConfiguredRange) {
    DeviceBuffer<Vector3<double>> values(64);

    UniformGenerateOperator<double> {}.generate(values, -2.0, 3.0, 7u);

    EXPECT_TRUE(test::all_finite_points(values));
    EXPECT_TRUE(test::points_in_range(values, -2.0, 3.0));
}

TEST(UniformGenerateOperator, GenerateWithSameSeedProducesSameSequence) {
    DeviceBuffer<Vector3<double>> lhs(32);
    DeviceBuffer<Vector3<double>> rhs(32);

    UniformGenerateOperator<double> {}.generate(lhs, -1.0, 1.0, 11u);
    UniformGenerateOperator<double> {}.generate(rhs, -1.0, 1.0, 11u);

    EXPECT_TRUE(test::point_buffers_near(lhs, rhs, eps));
}

TEST(UniformGenerateOperator, GenerateLeavesEmptyBufferEmpty) {
    DeviceBuffer<Vector3<double>> values;

    UniformGenerateOperator<double> {}.generate(values, -1.0, 1.0, 3u);

    EXPECT_TRUE(values.empty());
}
