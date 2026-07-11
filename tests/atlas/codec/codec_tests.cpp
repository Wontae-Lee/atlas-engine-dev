#include <atlas/codec/codec.h>

#include <atlas/codec/codec_type.h>
#include <atlas/codec/knudsen_codec.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace {

using atlas::Codec;
using atlas::CodecType;
using atlas::CodecVariant;
using atlas::KnudsenCodec;
using atlas::SQRT_TWO;

// Reproduce KnudsenCodec::knudsen_number in the header's float order to check that a
// leaf's representative scalars survived a move.
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

// A leaf whose statistical weight differs from the default, so knudsen_number() gives a
// value distinguishable from a default-constructed leaf.
KnudsenCodec
make_weighted_leaf() {
    return KnudsenCodec::builder().with_representative_statistical_weight(4.0f).build();
}

}

// The umbrella is move-only: copy construction and copy assignment are deleted.
static_assert(!std::is_copy_constructible_v<Codec>, "Codec must be move-only");
static_assert(!std::is_copy_assignable_v<Codec>, "Codec must be move-only");
static_assert(std::is_move_constructible_v<Codec>, "Codec must be move-constructible");
static_assert(std::is_move_assignable_v<Codec>, "Codec must be move-assignable");

TEST(Codec, DefaultConstructsKnudsen) {
    const Codec codec {};

    EXPECT_EQ(codec.type, CodecType::knudsen);
}

TEST(Codec, WrapsKnudsenLeaf) {
    const Codec codec(make_weighted_leaf());

    EXPECT_EQ(codec.type, CodecType::knudsen);
    EXPECT_FLOAT_EQ(codec.knudsen.knudsen_number(2.0f),
                    reference_kn(2.0f, 1.0f, 1.0f, 4.0f, 1.0f));
}

TEST(Codec, AllocateIsNoOpOnNullBuffers) {
    const Codec codec {};

    // With null input/output the leaf returns before touching the device, so the
    // umbrella dispatch is safe to exercise on the host.
    codec.allocate(nullptr, nullptr, nullptr);

    SUCCEED();
}

TEST(Codec, MoveConstructPreservesLeafState) {
    Codec       source(make_weighted_leaf());
    const Codec moved = std::move(source);

    EXPECT_EQ(moved.type, CodecType::knudsen);
    EXPECT_FLOAT_EQ(moved.knudsen.knudsen_number(2.0f),
                    reference_kn(2.0f, 1.0f, 1.0f, 4.0f, 1.0f));
}

TEST(Codec, MoveAssignReplacesLeafState) {
    Codec codec {}; // Default leaf: statistical weight 1.0.
    ASSERT_FLOAT_EQ(codec.knudsen.knudsen_number(2.0f),
                    reference_kn(2.0f, 1.0f, 1.0f, 1.0f, 1.0f));

    Codec source(make_weighted_leaf());
    codec = std::move(source);

    EXPECT_EQ(codec.type, CodecType::knudsen);
    EXPECT_FLOAT_EQ(codec.knudsen.knudsen_number(2.0f),
                    reference_kn(2.0f, 1.0f, 1.0f, 4.0f, 1.0f));
}

TEST(Codec, SelfMoveAssignPreservesLeaf) {
    Codec codec(make_weighted_leaf());

    // Call the dispatcher directly to exercise the self-move guard without tripping the
    // compiler's self-move-in-assignment diagnostic: nothing is destroyed or rebuilt,
    // so the leaf's representative scalars stay intact.
    CodecVariant::move_assign(codec, std::move(codec));

    EXPECT_EQ(codec.type, CodecType::knudsen);
    EXPECT_FLOAT_EQ(codec.knudsen.knudsen_number(2.0f),
                    reference_kn(2.0f, 1.0f, 1.0f, 4.0f, 1.0f));
}

TEST(CodecType, RoundTripsThroughUnderlyingInt) {
    static_assert(std::is_same_v<std::underlying_type_t<CodecType>, int>,
                  "CodecType must have a fixed int underlying type for device buffers");

    const auto value = static_cast<int>(CodecType::knudsen);

    EXPECT_EQ(static_cast<CodecType>(value), CodecType::knudsen);
}
