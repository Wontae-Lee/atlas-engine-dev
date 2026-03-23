#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(UniformGenerateOperator, GenerateProducesValuesInsideConfiguredRange) {
    UniformGenerateOperator<double> op(7u);
    const auto value = op.generate(-2.0, 3.0);

    EXPECT_TRUE(test::is_finite_vec(value));
    EXPECT_GE(value.x, -2.0);
    EXPECT_LT(value.x, 3.0);
    EXPECT_GE(value.y, -2.0);
    EXPECT_LT(value.y, 3.0);
    EXPECT_GE(value.z, -2.0);
    EXPECT_LT(value.z, 3.0);
}

TEST(UniformGenerateOperator, GenerateWithSameSeedProducesSameSequence) {
    UniformGenerateOperator<double> lhs_op(11u);
    UniformGenerateOperator<double> rhs_op(11u);
    const auto lhs = lhs_op.generate(-1.0, 1.0);
    const auto rhs = rhs_op.generate(-1.0, 1.0);

    EXPECT_TRUE(test::vec_near(lhs, rhs, eps));
}

TEST(UniformGenerateOperator, ConsecutiveGenerateCallsAdvanceSequence) {
    UniformGenerateOperator<double> op(3u);
    const auto lhs = op.generate(-1.0, 1.0);
    const auto rhs = op.generate(-1.0, 1.0);

    EXPECT_FALSE(test::vec_near(lhs, rhs, eps));
}