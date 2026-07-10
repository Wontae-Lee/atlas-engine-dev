#include <atlas/random/uniform_real_distribution.h>

#include <gtest/gtest.h>

namespace {

using atlas::default_random_engine;
using atlas::uniform_real_distribution;

constexpr default_random_engine::result_type reference_seed = 20240710u;

}

TEST(UniformRealDistribution, DefaultRangeIsUnitInterval) {
    const uniform_real_distribution<float> dist;
    EXPECT_FLOAT_EQ(dist.min(), 0.0f);
    EXPECT_FLOAT_EQ(dist.max(), 1.0f);
}

TEST(UniformRealDistribution, BoundsAccessorsEchoConstruction) {
    const uniform_real_distribution<float> dist(2.0f, 7.0f);
    EXPECT_FLOAT_EQ(dist.min(), 2.0f);
    EXPECT_FLOAT_EQ(dist.max(), 7.0f);
}

TEST(UniformRealDistribution, DrawsStayInHalfOpenUnitInterval) {
    default_random_engine engine(reference_seed);
    const uniform_real_distribution<float> dist;

    for (int i = 0; i < 8192; ++i) {
        const float u = dist(engine);
        EXPECT_GE(u, 0.0f);
        // The upper bound is exclusive: the implementation nudges a rounded-up
        // canonical variate back below one.
        EXPECT_LT(u, 1.0f);
    }
}

TEST(UniformRealDistribution, CustomRangeStaysWithinBounds) {
    default_random_engine engine(reference_seed);
    const float lo = -3.0f;
    const float hi = 5.0f;
    const uniform_real_distribution<float> dist(lo, hi);

    for (int i = 0; i < 8192; ++i) {
        const float u = dist(engine);
        EXPECT_GE(u, lo);
        EXPECT_LT(u, hi);
    }
}

TEST(UniformRealDistribution, DegenerateRangeReturnsTheBound) {
    default_random_engine engine(reference_seed);
    const uniform_real_distribution<float> dist(3.0f, 3.0f);

    // With lo == hi the scale (hi - lo) is zero, so every draw is exactly lo.
    for (int i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(dist(engine), 3.0f);
    }
}

TEST(UniformRealDistribution, SampleMeanIsNearMidpoint) {
    default_random_engine engine(reference_seed);
    const float lo = 2.0f;
    const float hi = 10.0f;
    const uniform_real_distribution<float> dist(lo, hi);

    const int count = 20000;
    double sum      = 0.0;
    for (int i = 0; i < count; ++i) {
        sum += dist(engine);
    }

    // Standard error over 20k draws is ~0.016; the band is far wider so a fixed
    // seed cannot flake it.
    const double mean     = sum / static_cast<double>(count);
    const double midpoint = 0.5 * (static_cast<double>(lo) + static_cast<double>(hi));
    EXPECT_NEAR(mean, midpoint, 0.25);
}
