#include <atlas/codec/knudsen_codec.h>

#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

using atlas::KnudsenCodec;
using atlas::SQRT_TWO;

// Mirror of KnudsenCodec::knudsen_number, in the exact float order the header uses,
// so expected values track the implementation rather than a hand-computed constant.
float
reference_kn(const float particle_count,
             const float length,
             const float cross,
             const float weight,
             const float volume) {
    const float number_density = particle_count * weight / volume;
    const float mean_free_path = 1.0f / (SQRT_TWO * number_density * cross);
    return mean_free_path / length;
}

}

// The leaf is captured by value inside the device lambda, so it must stay trivially copyable.
static_assert(std::is_trivially_copyable_v<KnudsenCodec>,
              "KnudsenCodec must be trivially copyable for device capture");

TEST(KnudsenCodecBuilder, BuildSucceedsWithPositiveScalars) {
    const KnudsenCodec codec = KnudsenCodec::builder()
                                   .with_representative_characteristic_length(2.0f)
                                   .with_representative_collision_cross_sectional_area(3.0f)
                                   .with_representative_statistical_weight(4.0f)
                                   .with_representative_cell_volume(5.0f)
                                   .build();

    // Verify the configured scalars survived by reproducing the Knudsen formula.
    EXPECT_FLOAT_EQ(codec.knudsen_number(10.0f), reference_kn(10.0f, 2.0f, 3.0f, 4.0f, 5.0f));
}

TEST(KnudsenCodecBuilder, RejectsNonPositiveCharacteristicLength) {
    EXPECT_THROW(
        static_cast<void>(
            KnudsenCodec::builder().with_representative_characteristic_length(0.0f).build()),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(
            KnudsenCodec::builder().with_representative_characteristic_length(-1.0f).build()),
        std::invalid_argument);
}

TEST(KnudsenCodecBuilder, RejectsNonPositiveCrossSection) {
    EXPECT_THROW(
        static_cast<void>(
            KnudsenCodec::builder().with_representative_collision_cross_sectional_area(0.0f).build()),
        std::invalid_argument);
}

TEST(KnudsenCodecBuilder, RejectsNonPositiveStatisticalWeight) {
    EXPECT_THROW(
        static_cast<void>(
            KnudsenCodec::builder().with_representative_statistical_weight(-0.5f).build()),
        std::invalid_argument);
}

TEST(KnudsenCodecBuilder, RejectsNonPositiveCellVolume) {
    EXPECT_THROW(
        static_cast<void>(
            KnudsenCodec::builder().with_representative_cell_volume(0.0f).build()),
        std::invalid_argument);
}

TEST(KnudsenCodecBuilder, MakeHostSharedBuildsLeaf) {
    const auto codec = KnudsenCodec::builder()
                           .with_representative_statistical_weight(2.0f)
                           .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(codec));
    EXPECT_FLOAT_EQ(codec->knudsen_number(4.0f), reference_kn(4.0f, 1.0f, 1.0f, 2.0f, 1.0f));
}

TEST(KnudsenCodecKnudsenNumber, DefaultLeafMatchesFormula) {
    const KnudsenCodec codec {}; // All representative scalars default to 1.0.

    EXPECT_FLOAT_EQ(codec.knudsen_number(2.0f), reference_kn(2.0f, 1.0f, 1.0f, 1.0f, 1.0f));
}

TEST(KnudsenCodecKnudsenNumber, ZeroParticleCountYieldsZero) {
    const KnudsenCodec codec {};

    // Zero particles -> zero number density -> the guarded fallback of exactly 0.
    EXPECT_FLOAT_EQ(codec.knudsen_number(0.0f), 0.0f);
}

TEST(KnudsenCodecKnudsenNumber, NonPositiveCellVolumeYieldsZero) {
    const KnudsenCodec codec(1.0f, 1.0f, 1.0f, 0.0f); // volume == 0 -> number density 0.

    EXPECT_FLOAT_EQ(codec.knudsen_number(5.0f), 0.0f);
}

TEST(KnudsenCodecKnudsenNumber, NonPositiveCrossSectionYieldsZero) {
    const KnudsenCodec codec(1.0f, 0.0f, 1.0f, 1.0f); // cross-section guard trips.

    EXPECT_FLOAT_EQ(codec.knudsen_number(5.0f), 0.0f);
}

TEST(KnudsenCodecKnudsenNumber, NonPositiveCharacteristicLengthYieldsZero) {
    const KnudsenCodec codec(0.0f, 1.0f, 1.0f, 1.0f); // length guard trips.

    EXPECT_FLOAT_EQ(codec.knudsen_number(5.0f), 0.0f);
}

TEST(KnudsenCodecKnudsenNumber, NanParticleCountYieldsZero) {
    const KnudsenCodec codec {};

    // A NaN particle count makes the number density NaN; the `!(number_density > 0)`
    // guard rejects it (a NaN comparison is false), so the fallback of exactly 0 wins.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FLOAT_EQ(codec.knudsen_number(nan), 0.0f);
}

// Split table is fixed at {0.01, 0.1, 1.0, 10.0}; solver_index counts thresholds the
// value meets or exceeds, so a value exactly at a threshold buckets into the higher index.

TEST(KnudsenCodecSolverIndex, BelowFirstSplitIsZero) {
    const KnudsenCodec codec {};

    EXPECT_EQ(codec.solver_index(0.0f), 0);
    EXPECT_EQ(codec.solver_index(0.009f), 0);
}

TEST(KnudsenCodecSolverIndex, FirstThresholdBucketsUp) {
    const KnudsenCodec codec {};

    // Just below the 0.01 split stays at 0; exactly at 0.01 moves to 1.
    EXPECT_EQ(codec.solver_index(0.0099f), 0);
    EXPECT_EQ(codec.solver_index(0.01f), 1);
    EXPECT_EQ(codec.solver_index(0.05f), 1);
}

TEST(KnudsenCodecSolverIndex, SecondThresholdBucketsUp) {
    const KnudsenCodec codec {};

    EXPECT_EQ(codec.solver_index(0.099f), 1);
    EXPECT_EQ(codec.solver_index(0.1f), 2);
    EXPECT_EQ(codec.solver_index(0.5f), 2);
}

TEST(KnudsenCodecSolverIndex, ThirdThresholdBucketsUp) {
    const KnudsenCodec codec {};

    EXPECT_EQ(codec.solver_index(0.99f), 2);
    EXPECT_EQ(codec.solver_index(1.0f), 3);
    EXPECT_EQ(codec.solver_index(5.0f), 3);
}

TEST(KnudsenCodecSolverIndex, LastThresholdSaturatesAtSplitCount) {
    const KnudsenCodec codec {};

    // At or above the final 10.0 split, the index saturates at split_count (4).
    EXPECT_EQ(codec.solver_index(9.99f), 3);
    EXPECT_EQ(codec.solver_index(10.0f), KnudsenCodec::split_count);
    EXPECT_EQ(codec.solver_index(100.0f), KnudsenCodec::split_count);
}

TEST(KnudsenCodecSolverIndex, NanSaturatesToSplitCount) {
    const KnudsenCodec codec {};

    // The bucketing test is `!(kn < split)`, which is TRUE for a NaN kn (every NaN
    // comparison is false), so the loop advances at each threshold and saturates at
    // split_count rather than stopping at 0.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(codec.solver_index(nan), KnudsenCodec::split_count);
}
