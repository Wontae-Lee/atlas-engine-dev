#include <atlas/math/constants.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace {

using atlas::isfinite;

}

TEST(Constants, PiMatchesFourArctangentOfOne) {
    EXPECT_NEAR(atlas::pi, 4.0f * std::atan(1.0f), 1.0e-6f);
}

TEST(Constants, SqrtTwoSquaresToTwo) {
    EXPECT_NEAR(atlas::SQRT_TWO * atlas::SQRT_TWO, 2.0f, 1.0e-6f);
}

TEST(Constants, EpsilonAndToleranceAreEqual) {
    EXPECT_FLOAT_EQ(atlas::eps, atlas::tol);
}

TEST(Constants, FarIsAFiniteStandInBelowInfinity) {
    EXPECT_TRUE(atlas::isfinite(atlas::far));
    EXPECT_LT(atlas::far, atlas::inf);
}

TEST(Constants, InfinityIsNotFinite) {
    EXPECT_TRUE(std::isinf(atlas::inf));
    EXPECT_FALSE(atlas::isfinite(atlas::inf));
}

TEST(Constants, PhysicalConstantsArePositiveAndFinite) {
    EXPECT_TRUE(atlas::isfinite(atlas::gravity));
    EXPECT_GT(atlas::gravity, 0.0f);
    EXPECT_TRUE(atlas::isfinite(atlas::boltzmann_constant));
    EXPECT_GT(atlas::boltzmann_constant, 0.0f);
}

TEST(IsFinite, RejectsInfinityAndNaN) {
    EXPECT_TRUE(atlas::isfinite(1.0f));
    EXPECT_TRUE(atlas::isfinite(0.0f));
    EXPECT_FALSE(atlas::isfinite(std::numeric_limits<float>::infinity()));
    EXPECT_FALSE(atlas::isfinite(std::numeric_limits<float>::quiet_NaN()));
}

TEST(SqrtNonnegative, ReturnsSquareRootForPositiveInput) {
    EXPECT_FLOAT_EQ(atlas::sqrt_nonnegative(4.0f), 2.0f);
}

TEST(SqrtNonnegative, ClampsNonPositiveInputToZero) {
    EXPECT_FLOAT_EQ(atlas::sqrt_nonnegative(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(atlas::sqrt_nonnegative(-9.0f), 0.0f);
}

TEST(SolveQuadratic, ReturnsRealRootsSorted) {
    // t^2 - 3t + 2 = 0 has roots 1 and 2.
    float t0 = 0.0f;
    float t1 = 0.0f;
    EXPECT_TRUE(atlas::solve_quadratic(1.0f, -3.0f, 2.0f, t0, t1));
    EXPECT_NEAR(t0, 1.0f, 1.0e-6f);
    EXPECT_NEAR(t1, 2.0f, 1.0e-6f);
}

TEST(SolveQuadratic, ReturnsFalseAndLeavesRootsUntouchedForNegativeDiscriminant) {
    // t^2 + 1 = 0 has no real root.
    float t0 = -11.0f;
    float t1 = -22.0f;
    EXPECT_FALSE(atlas::solve_quadratic(1.0f, 0.0f, 1.0f, t0, t1));
    EXPECT_FLOAT_EQ(t0, -11.0f);
    EXPECT_FLOAT_EQ(t1, -22.0f);
}
