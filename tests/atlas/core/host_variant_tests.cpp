#include <atlas/core/host_variant.h>

#include <atlas/core/macros.h>

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace {

using atlas::HostVariant;
using atlas::HostVariantCase;

// Discriminant for the dummy umbrella. `unknown` is deliberately left out of the
// variant's case list so it exercises the normalize/fallback paths.
enum class Kind { alpha, beta, gamma, unknown };

// Process-wide tallies of leaf lifecycle events. The dummy leaves bump these on
// construction, move and destruction so the tests can assert precisely how many moves and
// destructor calls a HostVariant operation performed — the move-only resource safety this
// variant exists to provide. Reset explicitly at the start of any test that inspects a delta.
struct LeafStats final {
    int value_ctors = 0; // Value (heap-allocating) constructions.
    int move_ctors  = 0; // Move constructions (must be used, never a copy).
    int destructors = 0; // Destructor calls.
    int live        = 0; // Currently-alive leaf objects (0 means no leak / no double free).

    void reset() noexcept { *this = LeafStats {}; }
};

LeafStats stats;

// Shared machinery for the move-only dummy leaves. Owns a single heap `int` as a stand-in
// for the `DeviceBuffer` a real leaf would hold, which makes the leaf genuinely move-only:
// the copy operations are deleted, so a copy is a compile error rather than something a
// runtime counter has to catch. Every lifecycle event updates `stats` so the tests can prove
// a move (not a copy) was used and that each active leaf is destroyed exactly once. A
// moved-from leaf holds a null `resource` — the valid, documented state HostVariant leaves
// the source in.
struct MoveOnlyLeaf {
    int* resource = nullptr;

    explicit MoveOnlyLeaf(const int v) : resource(new int(v)) {
        ++stats.value_ctors;
        ++stats.live;
    }

    MoveOnlyLeaf(MoveOnlyLeaf&& other) noexcept : resource(other.resource) {
        other.resource = nullptr;
        ++stats.move_ctors;
        ++stats.live;
    }

    // Copy is deleted: a DeviceBuffer-owning leaf cannot be copied.
    MoveOnlyLeaf(const MoveOnlyLeaf&)            = delete;
    MoveOnlyLeaf& operator=(const MoveOnlyLeaf&) = delete;

    ~MoveOnlyLeaf() {
        delete resource;
        ++stats.destructors;
        --stats.live;
    }

    // Reads the owned payload; only valid while the leaf still holds its resource.
    int value() const noexcept { return *resource; }

    // True while this leaf still owns its resource (false once moved from).
    bool has_resource() const noexcept { return resource != nullptr; }

    // Mutates the owned payload in place.
    void bump() noexcept { ++(*resource); }
};

// Move-only leaf carrying compile-time id 1 (so a visitor can report which leaf it hit).
struct Alpha final : MoveOnlyLeaf {
    static constexpr int id = 1;

    explicit Alpha(const int v = 0) : MoveOnlyLeaf(v) {}
};

// Move-only leaf carrying compile-time id 2.
struct Beta final : MoveOnlyLeaf {
    static constexpr int id = 2;

    explicit Beta(const int v = 0) : MoveOnlyLeaf(v) {}
};

// Move-only leaf carrying compile-time id 3.
struct Gamma final : MoveOnlyLeaf {
    static constexpr int id = 3;

    explicit Gamma(const int v = 0) : MoveOnlyLeaf(v) {}
};

// Visitor returning the active leaf's compile-time id (proves which leaf was hit).
struct ReadId final {
    template <typename L>
    int
    operator()(const L&) const noexcept { return L::id; }
};

// Visitor returning the active leaf's stored value.
struct ReadValue final {
    template <typename L>
    int
    operator()(const L& leaf) const noexcept { return leaf.value(); }
};

// Visitor reporting whether the active leaf still owns its resource.
struct HasResource final {
    template <typename L>
    bool
    operator()(const L& leaf) const noexcept { return leaf.has_resource(); }
};

// Mutating visitor that bumps the active leaf's value in place.
struct Bump final {
    template <typename L>
    void
    operator()(L& leaf) const noexcept { leaf.bump(); }
};

// Const visitor copying the active leaf's value into a caller-owned slot.
struct CaptureValue final {
    int* out;

    template <typename L>
    void
    operator()(const L& leaf) const noexcept { *out = leaf.value(); }
};

// Minimal move-only umbrella mirroring the buffer-owning leaf pattern. Because the leaves
// own a heap resource the union's special members are deleted, so the umbrella hands its
// whole lifecycle to HostVariant: default/payload construction, move-construct, move-assign
// and destroy. Copy is deleted, matching a real Source-style umbrella that owns a
// `DeviceBuffer`.
class Shape final {
public:
    Kind type = Kind::alpha;

    union {
        Alpha alpha;
        Beta  beta;
        Gamma gamma;
    };

    Shape();

    // Move-only: buffer-owning leaves cannot be copied.
    Shape(const Shape&)            = delete;
    Shape& operator=(const Shape&) = delete;

    Shape(Shape&& other) noexcept;

    Shape&
    operator=(Shape&& other) noexcept;

    ~Shape();

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Shape>, int> = 0>
    explicit Shape(Payload&& payload);
};

// Declared once Shape is complete: forming `&Shape::alpha` needs the full type, so every
// member that reaches back into the variant is defined out of line.
using ShapeVariant = HostVariant<
    Shape,
    Kind,
    Kind::alpha,
    HostVariantCase<Kind::alpha, &Shape::alpha>,
    HostVariantCase<Kind::beta, &Shape::beta>,
    HostVariantCase<Kind::gamma, &Shape::gamma>>;

inline Shape::Shape() {
    ShapeVariant::construct(*this, Kind::alpha);
}

inline Shape::Shape(Shape&& other) noexcept {
    ShapeVariant::move_construct(*this, std::move(other));
}

inline Shape&
Shape::operator=(Shape&& other) noexcept {
    ShapeVariant::move_assign(*this, std::move(other));
    return *this;
}

inline Shape::~Shape() {
    ShapeVariant::destroy(*this);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Shape>, int> Enable>
Shape::Shape(Payload&& payload) {
    ShapeVariant::construct_payload(*this, std::forward<Payload>(payload));
}

} // namespace

// HostVariant exists precisely for buffer-owning, move-only umbrellas: the opposite of
// the trivially-copyable contract DeviceVariant guarantees.
static_assert(!std::is_trivially_copyable_v<Shape>,
              "Shape owns a heap resource and must not be trivially copyable");
static_assert(!std::is_copy_constructible_v<Shape>, "the umbrella must be move-only");
static_assert(!std::is_copy_assignable_v<Shape>, "the umbrella must be move-only");
static_assert(std::is_move_constructible_v<Shape>);
static_assert(std::is_move_assignable_v<Shape>);
static_assert(std::is_nothrow_move_constructible_v<Shape>,
              "move construction must not throw so the umbrella stays relocatable");

// The leaves themselves are move-only, which is the whole reason this variant exists.
static_assert(!std::is_copy_constructible_v<Alpha>, "leaves are move-only");
static_assert(std::is_nothrow_move_constructible_v<Alpha>);

TEST(HostVariant, DefaultConstructsFirstAlternative) {
    const Shape shape {};

    EXPECT_EQ(shape.type, Kind::alpha);
    EXPECT_EQ(ShapeVariant::visit(shape, ReadValue {}, -1), 0);
}

TEST(HostVariant, ConstructFromPayloadSetsMatchingTag) {
    const Shape alpha(Alpha(4));
    const Shape beta(Beta(5));
    const Shape gamma(Gamma(6));

    EXPECT_EQ(alpha.type, Kind::alpha);
    EXPECT_EQ(beta.type, Kind::beta);
    EXPECT_EQ(gamma.type, Kind::gamma);

    EXPECT_EQ(ShapeVariant::visit(gamma, ReadValue {}, -1), 6);
}

TEST(HostVariant, VisitDispatchesToActiveLeaf) {
    const Shape shape(Gamma(7));

    EXPECT_EQ(ShapeVariant::visit(shape, ReadId {}, -1), Gamma::id);
    EXPECT_EQ(ShapeVariant::visit(shape, ReadValue {}, -1), 7);
}

TEST(HostVariant, VisitReturnsFallbackForUnmatchedTag) {
    Shape shape(Beta(2));
    shape.type = Kind::unknown; // simulate a corrupt discriminant

    EXPECT_EQ(ShapeVariant::visit(shape, ReadValue {}, -7), -7);

    shape.type = Kind::beta; // restore so the destructor finds the live leaf
}

TEST(HostVariant, ApplyMutatesActiveLeaf) {
    Shape shape(Beta(10));

    ShapeVariant::apply(shape, Bump {});

    EXPECT_EQ(shape.type, Kind::beta);
    EXPECT_EQ(ShapeVariant::visit(shape, ReadValue {}, -1), 11);
}

TEST(HostVariant, ConstApplyReadsActiveLeaf) {
    const Shape shape(Alpha(3));

    int seen = -1;
    ShapeVariant::apply(shape, CaptureValue { &seen });

    EXPECT_EQ(seen, 3);
}

TEST(HostVariant, ContainsRecognizesOnlyRegisteredTags) {
    EXPECT_TRUE(ShapeVariant::contains(Kind::alpha));
    EXPECT_TRUE(ShapeVariant::contains(Kind::gamma));
    EXPECT_FALSE(ShapeVariant::contains(Kind::unknown));
}

TEST(HostVariant, NormalizeCollapsesUnknownTagToDefault) {
    EXPECT_EQ(ShapeVariant::normalize(Kind::beta), Kind::beta);
    EXPECT_EQ(ShapeVariant::normalize(Kind::unknown), Kind::alpha);
}

TEST(HostVariant, ConstructNormalizesUnknownTagToDefault) {
    Shape shape {};

    // Release the default member before re-initializing: unlike DeviceVariant these leaves
    // own a resource, so construct (which assumes no active member) must follow a destroy.
    ShapeVariant::destroy(shape);
    ShapeVariant::construct(shape, Kind::unknown, 9);

    EXPECT_EQ(shape.type, Kind::alpha);
    EXPECT_EQ(ShapeVariant::visit(shape, ReadValue {}, -1), 9);
}

TEST(HostVariant, IsMoveOnly) {
    EXPECT_FALSE(std::is_copy_constructible_v<Shape>);
    EXPECT_FALSE(std::is_copy_assignable_v<Shape>);
    EXPECT_TRUE(std::is_move_constructible_v<Shape>);
    EXPECT_TRUE(std::is_move_assignable_v<Shape>);
}

TEST(HostVariant, MoveConstructTransfersResourceWithoutCopy) {
    Shape source(Gamma(7));

    stats.reset();
    Shape moved(std::move(source));

    // Exactly one move, and copies are impossible (the leaf's copy ctor is deleted).
    EXPECT_EQ(stats.move_ctors, 1);
    EXPECT_EQ(moved.type, Kind::gamma);
    EXPECT_EQ(ShapeVariant::visit(moved, ReadValue {}, -1), 7);

    // The moved-from variant keeps its tag but its leaf is emptied: the valid, documented
    // state HostVariant leaves the source in.
    EXPECT_EQ(source.type, Kind::gamma);
    EXPECT_FALSE(ShapeVariant::visit(source, HasResource {}, true));
}

TEST(HostVariant, MoveAssignDestroysOldLeafThenConstructsNew) {
    Shape destination(Alpha(1));
    Shape source(Gamma(9));

    stats.reset();
    destination = std::move(source);

    // The differing active leaf is destroyed before the incoming one is constructed.
    EXPECT_EQ(stats.destructors, 1);
    EXPECT_EQ(stats.move_ctors, 1);

    EXPECT_EQ(destination.type, Kind::gamma);
    EXPECT_EQ(ShapeVariant::visit(destination, ReadValue {}, -1), 9);
    EXPECT_FALSE(ShapeVariant::visit(source, HasResource {}, true));
}

TEST(HostVariant, MoveAssignSameTagReplacesResource) {
    Shape destination(Beta(1));
    Shape source(Beta(42));

    destination = std::move(source);

    EXPECT_EQ(destination.type, Kind::beta);
    EXPECT_EQ(ShapeVariant::visit(destination, ReadValue {}, -1), 42);
    EXPECT_FALSE(ShapeVariant::visit(source, HasResource {}, true));
}

TEST(HostVariant, SelfMoveAssignIsNoOp) {
    Shape shape(Gamma(42));

    stats.reset();
    // Call the dispatcher directly to exercise the self-move guard without tripping
    // the compiler's self-move-in-assignment diagnostic.
    ShapeVariant::move_assign(shape, std::move(shape));

    // Nothing destroyed, nothing reconstructed, resource intact: no double free.
    EXPECT_EQ(stats.destructors, 0);
    EXPECT_EQ(stats.move_ctors, 0);
    EXPECT_EQ(shape.type, Kind::gamma);
    EXPECT_EQ(ShapeVariant::visit(shape, ReadValue {}, -1), 42);
}

TEST(HostVariant, DestructorRunsActiveLeafDestructorExactlyOnce) {
    stats.reset();
    {
        const Shape shape(Beta(5));

        // The Beta temporary is moved into the union and then destroyed at the end of the
        // full expression, so one destructor has already run. `live` is what isolates the
        // umbrella's own leaf from that bookkeeping.
        EXPECT_EQ(stats.value_ctors, 1);
        EXPECT_EQ(stats.destructors, 1);
        EXPECT_EQ(stats.live, 1);
    }

    // Exactly one more: the active leaf. No leak, no double free.
    EXPECT_EQ(stats.destructors, 2);
    EXPECT_EQ(stats.live, 0);
}
