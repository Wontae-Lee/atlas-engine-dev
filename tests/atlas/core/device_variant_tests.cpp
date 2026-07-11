#include <atlas/core/device_variant.h>

#include <atlas/core/macros.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::DeviceTypeCase;
using atlas::DeviceTypeSwitch;
using atlas::DeviceVariant;
using atlas::DeviceVariantCase;
using atlas::type_tag;

/// Discriminant for the dummy umbrella. `unknown` is deliberately left out of the
/// variant's case list so it exercises the normalize/fallback paths.
enum class Kind { alpha, beta, gamma, unknown };

// Trivially-copyable leaves: each carries a distinct payload and a compile-time id
// so a visitor can report which concrete leaf it was dispatched to.
struct Alpha final {
    int value;
    static constexpr int id = 1;

    ATLAS_ALL_DEVICE explicit Alpha(const int v = 0) noexcept : value(v) {}
};

struct Beta final {
    int value;
    static constexpr int id = 2;

    ATLAS_ALL_DEVICE explicit Beta(const int v = 0) noexcept : value(v) {}
};

struct Gamma final {
    int value;
    static constexpr int id = 3;

    ATLAS_ALL_DEVICE explicit Gamma(const int v = 0) noexcept : value(v) {}
};

/// Visitor returning the active leaf's compile-time id (proves which leaf was hit).
struct ReadId final {
    template <typename L>
    ATLAS_ALL_DEVICE int
    operator()(const L&) const noexcept { return L::id; }
};

/// Visitor returning the active leaf's stored value.
struct ReadValue final {
    template <typename L>
    ATLAS_ALL_DEVICE int
    operator()(const L& leaf) const noexcept { return leaf.value; }
};

/// Mutating visitor that bumps the active leaf's value in place.
struct Bump final {
    template <typename L>
    ATLAS_ALL_DEVICE void
    operator()(L& leaf) const noexcept { leaf.value += 1; }
};

/// Const visitor copying the active leaf's value into a caller-owned slot. A functor
/// rather than a lambda, since nvcc rejects extended lambdas passed to device code.
struct CaptureValue final {
    int* out;

    template <typename L>
    ATLAS_ALL_DEVICE void
    operator()(const L& leaf) const noexcept { *out = leaf.value; }
};

/// Type-only visitor recovering the payload id from a `type_tag`.
struct TypeId final {
    template <typename P>
    ATLAS_ALL_DEVICE int
    operator()(type_tag<P>) const noexcept { return P::id; }
};

/// Minimal umbrella mirroring the tagged-union leaf pattern of the real modules.
class Figure final {
public:
    Kind type = Kind::alpha;

    union {
        Alpha alpha;
        Beta  beta;
        Gamma gamma;
    };

    ATLAS_ALL_DEVICE
    Figure() noexcept;

    ATLAS_ALL_DEVICE
    Figure(const Figure&) noexcept = default;

    ATLAS_ALL_DEVICE Figure&
    operator=(const Figure&) noexcept = default;

    ATLAS_ALL_DEVICE
    ~Figure() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Figure>, int> = 0>
    ATLAS_ALL_DEVICE explicit Figure(const Payload& payload) noexcept;
};

// Declared once Figure is complete: forming `&Figure::alpha` needs the full type, and the
// constructors that reach back into the variant are defined out of line for the same reason.
// The real umbrellas (DsmcKernel, Geometry) are laid out exactly this way.
using FigureVariant = DeviceVariant<
    Figure,
    Kind,
    Kind::alpha,
    DeviceVariantCase<Kind::alpha, &Figure::alpha>,
    DeviceVariantCase<Kind::beta, &Figure::beta>,
    DeviceVariantCase<Kind::gamma, &Figure::gamma>>;

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Figure::Figure() noexcept {
    FigureVariant::construct(*this, Kind::alpha);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Figure>, int> Enable>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Figure::Figure(const Payload& payload) noexcept {
    FigureVariant::construct_payload(*this, payload);
}

using FigureSwitch = DeviceTypeSwitch<
    Kind,
    Kind::alpha,
    DeviceTypeCase<Kind, Kind::alpha, Alpha>,
    DeviceTypeCase<Kind, Kind::beta, Beta>,
    DeviceTypeCase<Kind, Kind::gamma, Gamma>>;

} // namespace

// The whole point of DeviceVariant is device-capturable, byte-copyable umbrellas.
static_assert(std::is_trivially_copyable_v<Alpha>);
static_assert(std::is_trivially_copyable_v<Beta>);
static_assert(std::is_trivially_copyable_v<Gamma>);
static_assert(std::is_trivially_copyable_v<Figure>,
              "Figure must be trivially copyable for device buffers");

TEST(DeviceVariant, DefaultConstructsFirstAlternative) {
    const Figure figure {};

    EXPECT_EQ(figure.type, Kind::alpha);
    EXPECT_EQ(FigureVariant::visit(figure, ReadValue {}, -1), 0);
}

TEST(DeviceVariant, ConstructFromPayloadSetsMatchingTag) {
    const Figure alpha(Alpha(4));
    const Figure beta(Beta(5));
    const Figure gamma(Gamma(6));

    EXPECT_EQ(alpha.type, Kind::alpha);
    EXPECT_EQ(beta.type, Kind::beta);
    EXPECT_EQ(gamma.type, Kind::gamma);
}

TEST(DeviceVariant, VisitDispatchesToActiveLeaf) {
    const Figure figure(Gamma(7));

    EXPECT_EQ(FigureVariant::visit(figure, ReadId {}, -1), Gamma::id);
    EXPECT_EQ(FigureVariant::visit(figure, ReadValue {}, -1), 7);
}

TEST(DeviceVariant, VisitReturnsFallbackForUnmatchedTag) {
    Figure figure(Beta(2));
    figure.type = Kind::unknown; // simulate a corrupt discriminant

    EXPECT_EQ(FigureVariant::visit(figure, ReadValue {}, -7), -7);
}

TEST(DeviceVariant, ApplyMutatesActiveLeaf) {
    Figure figure(Beta(10));

    FigureVariant::apply(figure, Bump {});

    EXPECT_EQ(figure.type, Kind::beta);
    EXPECT_EQ(FigureVariant::visit(figure, ReadValue {}, -1), 11);
}

TEST(DeviceVariant, ConstApplyReadsActiveLeaf) {
    const Figure figure(Alpha(3));

    int seen = -1;
    FigureVariant::apply(figure, CaptureValue { &seen });

    EXPECT_EQ(seen, 3);
}

TEST(DeviceVariant, ContainsRecognizesOnlyRegisteredTags) {
    EXPECT_TRUE(FigureVariant::contains(Kind::alpha));
    EXPECT_TRUE(FigureVariant::contains(Kind::gamma));
    EXPECT_FALSE(FigureVariant::contains(Kind::unknown));
}

TEST(DeviceVariant, NormalizeCollapsesUnknownTagToDefault) {
    EXPECT_EQ(FigureVariant::normalize(Kind::beta), Kind::beta);
    EXPECT_EQ(FigureVariant::normalize(Kind::unknown), Kind::alpha);
}

TEST(DeviceVariant, ConstructNormalizesUnknownTagToDefault) {
    Figure figure {};

    // An out-of-range tag must still leave a valid, default-tagged member alive.
    FigureVariant::construct(figure, Kind::unknown, 9);

    EXPECT_EQ(figure.type, Kind::alpha);
    EXPECT_EQ(FigureVariant::visit(figure, ReadValue {}, -1), 9);
}

TEST(DeviceVariant, IsCopyConstructible) {
    EXPECT_TRUE(std::is_copy_constructible_v<Figure>);
    EXPECT_TRUE(std::is_copy_assignable_v<Figure>);
}

TEST(DeviceVariant, CopyConstructPreservesActiveLeaf) {
    const Figure original(Gamma(7));
    const Figure copy = original;

    EXPECT_EQ(copy.type, Kind::gamma);
    EXPECT_EQ(FigureVariant::visit(copy, ReadValue {}, -1), 7);
}

TEST(DeviceVariant, CopyAssignReplacesActiveLeaf) {
    Figure destination(Alpha(1));

    destination = Figure(Gamma(3));

    EXPECT_EQ(destination.type, Kind::gamma);
    EXPECT_EQ(FigureVariant::visit(destination, ReadValue {}, -1), 3);
}

TEST(DeviceVariant, AssignReplacesActiveLeafInPlace) {
    Figure       destination(Alpha(1));
    const Figure source(Gamma(9));

    FigureVariant::assign(destination, source);

    EXPECT_EQ(destination.type, Kind::gamma);
    EXPECT_EQ(FigureVariant::visit(destination, ReadValue {}, -1), 9);
}

TEST(DeviceVariant, VisitTypeDispatchesByTagAlone) {
    EXPECT_EQ(FigureVariant::visit_type(Kind::beta, TypeId {}, -1), Beta::id);
    EXPECT_EQ(FigureVariant::visit_type(Kind::unknown, TypeId {}, -1), -1);
}

TEST(DeviceVariant, CopyConstructInitializesTargetFromSource) {
    // Exercise the static copy_construct directly (distinct from the `= default`
    // copy constructor): it activates the source's leaf in an uninitialized target.
    Figure       target {};
    const Figure source(Gamma(9));

    FigureVariant::copy_construct(target, source);

    EXPECT_EQ(target.type, Kind::gamma);
    EXPECT_EQ(FigureVariant::visit(target, ReadValue {}, -1), 9);
}

TEST(DeviceVariant, CopyConstructNormalizesUnknownSourceTag) {
    // copy_construct normalizes the source's tag before dispatch — unlike the trivial
    // `= default` memcpy copy, an out-of-range source tag folds back to DefaultTag while
    // the still-live default-arm leaf is copied across.
    Figure source(Alpha(3));
    source.type = Kind::unknown; // corrupt the tag; the alpha member stays alive

    Figure target {};
    FigureVariant::copy_construct(target, source);

    EXPECT_EQ(target.type, Kind::alpha);
    EXPECT_EQ(FigureVariant::visit(target, ReadValue {}, -1), 3);
}

TEST(DeviceVariant, SelfAssignPreservesActiveLeaf) {
    // The self-assignment guard returns before destroy/copy, so the active leaf and its
    // value are left intact rather than destroyed and rebuilt from itself.
    Figure figure(Beta(5));

    FigureVariant::assign(figure, figure);

    EXPECT_EQ(figure.type, Kind::beta);
    EXPECT_EQ(FigureVariant::visit(figure, ReadValue {}, -1), 5);
}

TEST(DeviceTypeSwitch, HoldsReportsMembership) {
    EXPECT_TRUE(FigureSwitch::holds<Alpha>);
    EXPECT_TRUE(FigureSwitch::holds<Gamma>);
    EXPECT_FALSE(FigureSwitch::holds<int>);
}

TEST(DeviceTypeSwitch, TagOfReturnsTagOrDefault) {
    EXPECT_EQ(FigureSwitch::tag_of<Beta>(), Kind::beta);
    EXPECT_EQ(FigureSwitch::tag_of<int>(), Kind::alpha);
}

TEST(DeviceTypeSwitch, VisitDispatchesOnTag) {
    EXPECT_EQ(FigureSwitch::visit(Kind::gamma, TypeId {}, -1), Gamma::id);
    EXPECT_EQ(FigureSwitch::visit(Kind::unknown, TypeId {}, -1), -1);
}
