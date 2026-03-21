#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(MaxwellSigmaGenerateOperator, GenerateProducesFiniteVelocities) {
    MaxwellSigmaGenerateOperator<double> op(5u);
    const auto value = op.generate(2.5);

    EXPECT_TRUE(test::is_finite_vec(value));
}

TEST(MaxwellSigmaGenerateOperator, GenerateWithSameSeedProducesSameSequence) {
    MaxwellSigmaGenerateOperator<double> lhs_op(13u);
    MaxwellSigmaGenerateOperator<double> rhs_op(13u);
    const auto lhs = lhs_op.generate(1.25);
    const auto rhs = rhs_op.generate(1.25);

    EXPECT_TRUE(test::vec_near(lhs, rhs, eps));
}

TEST(MaxwellSigmaGenerateOperator, ZeroSigmaProducesZeroVelocity) {
    MaxwellSigmaGenerateOperator<double> op(2u);
    const auto value = op.generate(0.0);

    EXPECT_TRUE(test::vec_near(value, Vector3<double>(0.0, 0.0, 0.0), eps));
}
