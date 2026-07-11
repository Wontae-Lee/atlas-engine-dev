#include <atlas/sampling/sampling.h>

#include <atlas/math/constants.h>
#include <atlas/math/vector/float3.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using atlas::default_random_engine;
using atlas::Float3;
using atlas::uniform_real_distribution;

// A fixed seed so every statistical case is deterministic run to run.
constexpr default_random_engine::result_type sampling_seed = 1234567u;

// How many draws the statistical cases average over; large enough to be tight, fast enough not to drag.
constexpr int statistical_sample_count = 100000;

// Assert a direction is unit length within a float-trig tolerance.
void
expect_unit_length(const Float3& v) {
    EXPECT_NEAR(v.length(), 1.0f, 1e-4f);
}

}

/* ------------------------------------------------------------------ shuffle_key */

TEST(ShuffleKey, IsDeterministic) {
    // The hash is stateless: the same (index, seed) always yields the same key.
    EXPECT_EQ(atlas::shuffle_key(7, 42u), atlas::shuffle_key(7, 42u));
}

TEST(ShuffleKey, AdjacentIndicesDecorrelate) {
    // Good avalanche means neighboring indices map to different keys.
    const std::uint64_t a = atlas::shuffle_key(1000, 1u);
    const std::uint64_t b = atlas::shuffle_key(1001, 1u);
    const std::uint64_t c = atlas::shuffle_key(1000, 2u);

    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
}

/* -------------------------------------------------- generate_standard_normal(_pair) */

TEST(GenerateStandardNormal, SameSeedReproducesSequence) {
    default_random_engine a(sampling_seed);
    default_random_engine b(sampling_seed);

    for (int i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(atlas::generate_standard_normal(a), atlas::generate_standard_normal(b));
    }
}

TEST(GenerateStandardNormal, PairMatchesFirstOfWrapper) {
    // The single-value wrapper returns the first of the Box-Muller pair.
    default_random_engine wrapper(sampling_seed);
    default_random_engine pair(sampling_seed);

    const float single = atlas::generate_standard_normal(wrapper);

    float first {};
    float second {};
    atlas::generate_standard_normal_pair(pair, first, second);

    EXPECT_FLOAT_EQ(single, first);
}

TEST(GenerateStandardNormal, MatchesUnitGaussianStatistics) {
    default_random_engine engine(sampling_seed);

    double sum         = 0.0;
    double sum_squares = 0.0;
    int    n           = 0;

    for (int i = 0; i < statistical_sample_count / 2; ++i) {
        float first {};
        float second {};
        atlas::generate_standard_normal_pair(engine, first, second);
        sum += first + second;
        sum_squares += static_cast<double>(first) * first + static_cast<double>(second) * second;
        n += 2;
    }

    const double mean     = sum / n;
    const double variance = sum_squares / n - mean * mean;

    // N(0, 1): mean ~ 0, variance ~ 1. Tolerances carry ample margin for 100k draws.
    EXPECT_NEAR(mean, 0.0, 0.02);
    EXPECT_NEAR(variance, 1.0, 0.05);
}

/* ------------------------------------------------------------ sample_uniform_vector */

TEST(SampleUniformVector, ComponentsStayInRange) {
    default_random_engine engine(sampling_seed);

    for (int i = 0; i < 4096; ++i) {
        const Float3 v = atlas::sample_uniform_vector(engine, -2.0f, 5.0f);
        EXPECT_GE(v.x, -2.0f);
        EXPECT_LT(v.x, 5.0f);
        EXPECT_GE(v.y, -2.0f);
        EXPECT_LT(v.y, 5.0f);
        EXPECT_GE(v.z, -2.0f);
        EXPECT_LT(v.z, 5.0f);
    }
}

TEST(SampleUniformVector, MeanIsIntervalMidpoint) {
    default_random_engine engine(sampling_seed);

    Float3 accumulated(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < statistical_sample_count; ++i) {
        accumulated += atlas::sample_uniform_vector(engine, 0.0f, 1.0f);
    }
    const Float3 mean = accumulated / static_cast<float>(statistical_sample_count);

    // Uniform on [0, 1) per axis, so each mean tends to 0.5.
    EXPECT_NEAR(mean.x, 0.5f, 0.01f);
    EXPECT_NEAR(mean.y, 0.5f, 0.01f);
    EXPECT_NEAR(mean.z, 0.5f, 0.01f);
}

/* ------------------------------------------------------------- sample_normal_vector */

TEST(SampleNormalVector, MatchesIsotropicGaussianStatistics) {
    default_random_engine engine(sampling_seed);

    const float sigma = 2.0f;

    Float3 sum(0.0f, 0.0f, 0.0f);
    Float3 sum_squares(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < statistical_sample_count; ++i) {
        const Float3 v = atlas::sample_normal_vector(engine, sigma);
        sum += v;
        sum_squares += v * v;
    }

    const float n    = static_cast<float>(statistical_sample_count);
    const Float3 mean = sum / n;

    // Zero mean per axis, and per-axis variance ~ sigma^2 = 4.
    EXPECT_NEAR(mean.x, 0.0f, 0.05f);
    EXPECT_NEAR(mean.y, 0.0f, 0.05f);
    EXPECT_NEAR(mean.z, 0.0f, 0.05f);

    EXPECT_NEAR(sum_squares.x / n, sigma * sigma, 0.2f);
    EXPECT_NEAR(sum_squares.y / n, sigma * sigma, 0.2f);
    EXPECT_NEAR(sum_squares.z / n, sigma * sigma, 0.2f);
}

/* --------------------------------------------------------- build_orthonormal_basis */

TEST(BuildOrthonormalBasis, ProducesOrthonormalFrame) {
    const Float3 n = Float3(1.0f, 2.0f, 3.0f).normalized();

    Float3 t;
    Float3 b;
    atlas::build_orthonormal_basis(n, t, b);

    expect_unit_length(t);
    expect_unit_length(b);
    EXPECT_NEAR(t.dot(n), 0.0f, 1e-4f);
    EXPECT_NEAR(b.dot(n), 0.0f, 1e-4f);
    EXPECT_NEAR(t.dot(b), 0.0f, 1e-4f);
}

TEST(BuildOrthonormalBasis, DegenerateNormalUsesWorldAxisFallback) {
    // A zero normal cannot yield a stable tangent, so the world X/Y axes stand in.
    Float3 t;
    Float3 b;
    atlas::build_orthonormal_basis(Float3(0.0f, 0.0f, 0.0f), t, b);

    EXPECT_EQ(t, Float3(1.0f, 0.0f, 0.0f));
    EXPECT_EQ(b, Float3(0.0f, 1.0f, 0.0f));
}

/* -------------------------------------------------------- sample_uniform_hemisphere */

TEST(SampleUniformHemisphere, SamplesAreUnitAndInHemisphere) {
    default_random_engine                  engine(sampling_seed);
    uniform_real_distribution<float>       dist(0.0f, 1.0f);
    const Float3                           normal = Float3(0.2f, -0.5f, 1.0f).normalized();

    for (int i = 0; i < 4096; ++i) {
        const float  u1 = dist(engine);
        const float  u2 = dist(engine);
        const Float3 d  = atlas::sample_uniform_hemisphere(normal, u1, u2);
        expect_unit_length(d);
        EXPECT_GE(d.dot(normal), -1e-4f);
    }
}

TEST(SampleUniformHemisphere, MeanProjectionIsHalf) {
    default_random_engine            engine(sampling_seed);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    const Float3                     normal(0.0f, 0.0f, 1.0f);

    Float3 accumulated(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < statistical_sample_count; ++i) {
        const float u1 = dist(engine);
        const float u2 = dist(engine);
        accumulated += atlas::sample_uniform_hemisphere(normal, u1, u2);
    }
    const Float3 mean = accumulated / static_cast<float>(statistical_sample_count);

    // Solid-angle-uniform hemisphere: E[cos theta] = 1/2, azimuth averages out.
    EXPECT_NEAR(mean.z, 0.5f, 0.01f);
    EXPECT_NEAR(mean.x, 0.0f, 0.01f);
    EXPECT_NEAR(mean.y, 0.0f, 0.01f);
}

/* --------------------------------------------------------- sample_cosine_hemisphere */

TEST(SampleCosineHemisphere, SamplesAreUnitAndInHemisphere) {
    default_random_engine            engine(sampling_seed);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    const Float3                     normal = Float3(-0.7f, 0.1f, 0.9f).normalized();

    for (int i = 0; i < 4096; ++i) {
        const float  u1 = dist(engine);
        const float  u2 = dist(engine);
        const Float3 d  = atlas::sample_cosine_hemisphere(normal, u1, u2);
        expect_unit_length(d);
        EXPECT_GE(d.dot(normal), -1e-4f);
    }
}

TEST(SampleCosineHemisphere, MeanProjectionIsTwoThirds) {
    default_random_engine            engine(sampling_seed);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    const Float3                     normal(0.0f, 0.0f, 1.0f);

    Float3 accumulated(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < statistical_sample_count; ++i) {
        const float u1 = dist(engine);
        const float u2 = dist(engine);
        accumulated += atlas::sample_cosine_hemisphere(normal, u1, u2);
    }
    const Float3 mean = accumulated / static_cast<float>(statistical_sample_count);

    // Cosine-weighted hemisphere: E[cos theta] = 2/3, tangential mean ~ 0.
    EXPECT_NEAR(mean.z, 2.0f / 3.0f, 0.01f);
    EXPECT_NEAR(mean.x, 0.0f, 0.01f);
    EXPECT_NEAR(mean.y, 0.0f, 0.01f);
}

/* -------------------------------------------------------- sample_random_unit_vector */

TEST(SampleRandomUnitVector, SamplesAreUnitLength) {
    default_random_engine engine(sampling_seed);

    for (int i = 0; i < 4096; ++i) {
        expect_unit_length(atlas::sample_random_unit_vector(engine));
    }
}

TEST(SampleRandomUnitVector, SameSeedReproducesSequence) {
    default_random_engine a(sampling_seed);
    default_random_engine b(sampling_seed);

    for (int i = 0; i < 16; ++i) {
        const Float3 va = atlas::sample_random_unit_vector(a);
        const Float3 vb = atlas::sample_random_unit_vector(b);
        EXPECT_FLOAT_EQ(va.x, vb.x);
        EXPECT_FLOAT_EQ(va.y, vb.y);
        EXPECT_FLOAT_EQ(va.z, vb.z);
    }
}

TEST(SampleRandomUnitVector, MeanIsNearZeroOverSphere) {
    default_random_engine engine(sampling_seed);

    Float3 accumulated(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < statistical_sample_count; ++i) {
        accumulated += atlas::sample_random_unit_vector(engine);
    }
    const Float3 mean = accumulated / static_cast<float>(statistical_sample_count);

    // A uniform sphere has zero mean direction.
    EXPECT_NEAR(mean.x, 0.0f, 0.02f);
    EXPECT_NEAR(mean.y, 0.0f, 0.02f);
    EXPECT_NEAR(mean.z, 0.0f, 0.02f);
}

/* --------------------------------------------------- sample_directional_unit_vector */

TEST(SampleDirectionalUnitVector, SamplesAreUnitLength) {
    default_random_engine engine(sampling_seed);
    const Float3          axis(0.0f, 0.0f, 1.0f);

    for (int i = 0; i < 4096; ++i) {
        expect_unit_length(atlas::sample_directional_unit_vector(axis, 8.0f, engine));
    }
}

TEST(SampleDirectionalUnitVector, AlphaOneDegeneratesToUniformSphere) {
    default_random_engine engine(sampling_seed);
    const Float3          axis(0.0f, 0.0f, 1.0f);

    Float3 accumulated(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < statistical_sample_count; ++i) {
        accumulated += atlas::sample_directional_unit_vector(axis, 1.0f, engine);
    }
    const Float3 mean = accumulated / static_cast<float>(statistical_sample_count);

    // alpha == 1 gives cos theta = 2u - 1, the uniform sphere: mean ~ 0.
    EXPECT_NEAR(mean.z, 0.0f, 0.02f);
}

TEST(SampleDirectionalUnitVector, LargeAlphaClustersAroundAxis) {
    default_random_engine engine(sampling_seed);
    const Float3          axis(0.0f, 0.0f, 1.0f);
    const float           alpha = 9.0f;

    Float3 accumulated(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < statistical_sample_count; ++i) {
        accumulated += atlas::sample_directional_unit_vector(axis, alpha, engine);
    }
    const Float3 mean = accumulated / static_cast<float>(statistical_sample_count);

    // E[cos theta] = (alpha - 1) / (alpha + 1) = 0.8 for alpha = 9.
    EXPECT_NEAR(mean.z, (alpha - 1.0f) / (alpha + 1.0f), 0.02f);
}

TEST(SampleDirectionalUnitVector, NonPositiveAlphaCoercedToOne) {
    // A non-positive alpha is treated as 1, so the two engines must agree.
    default_random_engine coerced(sampling_seed);
    default_random_engine unit(sampling_seed);
    const Float3          axis(0.0f, 0.0f, 1.0f);

    const Float3 a = atlas::sample_directional_unit_vector(axis, -3.0f, coerced);
    const Float3 b = atlas::sample_directional_unit_vector(axis, 1.0f, unit);

    EXPECT_FLOAT_EQ(a.x, b.x);
    EXPECT_FLOAT_EQ(a.y, b.y);
    EXPECT_FLOAT_EQ(a.z, b.z);
}

/* ------------------------------------------------------------- sample_axis_count */

TEST(SampleAxisCount, CountsInclusiveGridPoints) {
    // floor((upper - lower) / spacing) + 1 counts both endpoints when they align.
    EXPECT_EQ(atlas::sample_axis_count(0.0f, 10.0f, 2.0f), 6);
    // The last point may not reach the upper bound.
    EXPECT_EQ(atlas::sample_axis_count(0.0f, 9.0f, 2.0f), 5);
}

TEST(SampleAxisCount, ZeroLengthExtentYieldsOne) {
    EXPECT_EQ(atlas::sample_axis_count(5.0f, 5.0f, 1.0f), 1);
}

TEST(SampleAxisCount, RejectsInvalidArguments) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    EXPECT_EQ(atlas::sample_axis_count(3.0f, 1.0f, 1.0f), 0);  // negative extent
    EXPECT_EQ(atlas::sample_axis_count(0.0f, 1.0f, 0.0f), 0);  // zero spacing
    EXPECT_EQ(atlas::sample_axis_count(0.0f, 1.0f, -1.0f), 0); // negative spacing
    EXPECT_EQ(atlas::sample_axis_count(0.0f, inf, 1.0f), 0);   // non-finite
    EXPECT_EQ(atlas::sample_axis_count(nan, 1.0f, 1.0f), 0);   // non-finite
}

/* ------------------------------------- sample_hashed_unit_interval (sine hash) */

TEST(SampleHashedUnitIntervalSine, StaysInUnitInterval) {
    for (int i = 0; i < 256; ++i) {
        const Float3 seed(static_cast<float>(i) * 0.37f, -static_cast<float>(i) * 1.1f, 0.5f);
        const float  value = atlas::sample_hashed_unit_interval(seed, 0.31f);
        EXPECT_GE(value, 0.0f);
        EXPECT_LT(value, 1.0f);
    }
}

TEST(SampleHashedUnitIntervalSine, IsDeterministicAndSaltSensitive) {
    const Float3 seed(1.5f, -2.5f, 3.5f);

    EXPECT_FLOAT_EQ(atlas::sample_hashed_unit_interval(seed, 0.31f),
                    atlas::sample_hashed_unit_interval(seed, 0.31f));
    EXPECT_NE(atlas::sample_hashed_unit_interval(seed, 0.31f),
              atlas::sample_hashed_unit_interval(seed, 1.73f));
}

/* ---------------------------------- sample_hashed_unit_interval (integer hash) */

TEST(SampleHashedUnitIntervalIndex, StaysInUnitInterval) {
    for (int i = 0; i < 4096; ++i) {
        const float value = atlas::sample_hashed_unit_interval(i, 7u);
        EXPECT_GE(value, 0.0f);
        EXPECT_LT(value, 1.0f);
    }
}

TEST(SampleHashedUnitIntervalIndex, IsDeterministic) {
    EXPECT_FLOAT_EQ(atlas::sample_hashed_unit_interval(1234, 99u),
                    atlas::sample_hashed_unit_interval(1234, 99u));
}

TEST(SampleHashedUnitIntervalIndex, MeanIsOneHalf) {
    double sum = 0.0;
    for (int i = 0; i < statistical_sample_count; ++i) {
        sum += atlas::sample_hashed_unit_interval(i, 2024u);
    }
    // A high-quality hash over many indices averages to the interval midpoint.
    EXPECT_NEAR(sum / statistical_sample_count, 0.5, 0.01);
}

/* --------------------------------------------------------- sample_hashed_index */

TEST(SampleHashedIndex, StaysWithinBound) {
    const int upper = 13;
    for (int i = 0; i < 4096; ++i) {
        const int value = atlas::sample_hashed_index(i, upper, 5u);
        EXPECT_GE(value, 0);
        EXPECT_LT(value, upper);
    }
}

TEST(SampleHashedIndex, IsDeterministic) {
    EXPECT_EQ(atlas::sample_hashed_index(77, 10, 3u), atlas::sample_hashed_index(77, 10, 3u));
}

TEST(SampleHashedIndex, NonPositiveBoundReturnsZero) {
    EXPECT_EQ(atlas::sample_hashed_index(5, 0, 1u), 0);
    EXPECT_EQ(atlas::sample_hashed_index(5, -4, 1u), 0);
}

/* -------------------------------------------------------- sample_weighted_index */

TEST(SampleWeightedIndex, StaysWithinCount) {
    default_random_engine engine(sampling_seed);
    const float           weights[] = { 0.1f, 0.2f, 0.3f, 0.4f };

    for (int i = 0; i < 4096; ++i) {
        const int index = atlas::sample_weighted_index(weights, 4, engine);
        EXPECT_GE(index, 0);
        EXPECT_LT(index, 4);
    }
}

TEST(SampleWeightedIndex, SingleWeightAlwaysSelectsIt) {
    default_random_engine engine(sampling_seed);
    const float           weights[] = { 1.0f };

    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(atlas::sample_weighted_index(weights, 1, engine), 0);
    }
}

TEST(SampleWeightedIndex, NonPositiveCountReturnsZero) {
    default_random_engine engine(sampling_seed);
    const float           weights[] = { 0.5f, 0.5f };

    EXPECT_EQ(atlas::sample_weighted_index(weights, 0, engine), 0);
    EXPECT_EQ(atlas::sample_weighted_index(weights, -2, engine), 0);
}

TEST(SampleWeightedIndex, IsDeterministic) {
    default_random_engine a(sampling_seed);
    default_random_engine b(sampling_seed);
    const float           weights[] = { 0.25f, 0.25f, 0.5f };

    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(atlas::sample_weighted_index(weights, 3, a),
                  atlas::sample_weighted_index(weights, 3, b));
    }
}

TEST(SampleWeightedIndex, FrequenciesTrackWeights) {
    default_random_engine engine(sampling_seed);
    const float           weights[] = { 0.25f, 0.75f };

    std::vector<int> hits(2, 0);
    for (int i = 0; i < statistical_sample_count; ++i) {
        ++hits[static_cast<std::size_t>(atlas::sample_weighted_index(weights, 2, engine))];
    }

    const double n = static_cast<double>(statistical_sample_count);
    EXPECT_NEAR(hits[0] / n, 0.25, 0.01);
    EXPECT_NEAR(hits[1] / n, 0.75, 0.01);
}

/* ------------------------------------------------------- sample_weighted_choice */

TEST(SampleWeightedChoice, ReturnsSelectedValue) {
    default_random_engine engine(sampling_seed);
    // The first weight is zero, so any draw lands on the second bucket's value.
    const float weights[] = { 0.0f, 1.0f };
    const float values[]  = { 5.0f, 9.0f };

    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(atlas::sample_weighted_choice(weights, values, 2, engine), std::size_t { 9 });
    }
}

TEST(SampleWeightedChoice, NonPositiveCountReturnsZero) {
    default_random_engine engine(sampling_seed);
    const float           weights[] = { 1.0f };
    const float           values[]  = { 42.0f };

    EXPECT_EQ(atlas::sample_weighted_choice(weights, values, 0, engine), std::size_t { 0 });
}
