#include <atlas/container/type_store.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

using atlas::TypeStore;

// Common base so several unrelated payloads can share one store.
struct State {
    virtual ~State() = default;
};

// A trivial host payload carrying an int tag.
struct AlphaState final : State {
    explicit AlphaState(const int tag) : tag(tag) { }
    int tag;
};

// A second, independent payload type keyed separately from AlphaState.
struct BetaState final : State {
    explicit BetaState(const double value) : value(value) { }
    double value;
};

}

TEST(TypeStore, DefaultConstructedIsEmpty) {
    const TypeStore<State> store {};

    EXPECT_TRUE(store.empty());
    EXPECT_EQ(store.size(), std::size_t { 0 });
}

TEST(TypeStore, EmplaceStoresAndGetReadsBack) {
    TypeStore<State> store {};

    AlphaState& stored = store.emplace<AlphaState>(42);
    EXPECT_EQ(stored.tag, 42);

    EXPECT_FALSE(store.empty());
    EXPECT_EQ(store.size(), std::size_t { 1 });

    const AlphaState* fetched = store.get<AlphaState>();
    ASSERT_NE(fetched, nullptr);
    EXPECT_EQ(fetched->tag, 42);
}

TEST(TypeStore, EmplaceReplacesExistingValueForSameType) {
    TypeStore<State> store {};

    store.emplace<AlphaState>(1);
    store.emplace<AlphaState>(2);

    // One instance per type: the second emplace overwrites the first.
    EXPECT_EQ(store.size(), std::size_t { 1 });
    ASSERT_NE(store.get<AlphaState>(), nullptr);
    EXPECT_EQ(store.get<AlphaState>()->tag, 2);
}

TEST(TypeStore, GetReturnsNullForAbsentType) {
    TypeStore<State> store {};
    store.emplace<AlphaState>(7);

    // A type that was never inserted yields nullptr rather than throwing.
    EXPECT_EQ(store.get<BetaState>(), nullptr);
}

TEST(TypeStore, SetTakesOwnershipOfValue) {
    TypeStore<State> store {};

    store.set(std::make_unique<BetaState>(3.5));

    const BetaState* fetched = store.get<BetaState>();
    ASSERT_NE(fetched, nullptr);
    EXPECT_DOUBLE_EQ(fetched->value, 3.5);
}

TEST(TypeStore, SetThrowsOnNullValue) {
    TypeStore<State> store {};

    EXPECT_THROW(store.set(std::unique_ptr<AlphaState> {}), std::invalid_argument);
}

TEST(TypeStore, DistinctTypesAreStoredIndependently) {
    TypeStore<State> store {};

    store.emplace<AlphaState>(10);
    store.emplace<BetaState>(2.0);

    EXPECT_EQ(store.size(), std::size_t { 2 });

    ASSERT_NE(store.get<AlphaState>(), nullptr);
    ASSERT_NE(store.get<BetaState>(), nullptr);
    EXPECT_EQ(store.get<AlphaState>()->tag, 10);
    EXPECT_DOUBLE_EQ(store.get<BetaState>()->value, 2.0);
}

TEST(TypeStore, ContainsReflectsPresence) {
    TypeStore<State> store {};

    EXPECT_FALSE(store.contains<AlphaState>());

    store.emplace<AlphaState>(1);

    EXPECT_TRUE(store.contains<AlphaState>());
    EXPECT_FALSE(store.contains<BetaState>());
}

TEST(TypeStore, RemoveDetachesValueAndShrinksStore) {
    TypeStore<State> store {};
    store.emplace<AlphaState>(99);

    std::unique_ptr<AlphaState> removed = store.remove<AlphaState>();

    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->tag, 99);
    EXPECT_FALSE(store.contains<AlphaState>());
    EXPECT_TRUE(store.empty());
}

TEST(TypeStore, RemoveReturnsNullWhenAbsent) {
    TypeStore<State> store {};

    EXPECT_EQ(store.remove<AlphaState>(), nullptr);
}

TEST(TypeStore, ClearRemovesEveryValue) {
    TypeStore<State> store {};
    store.emplace<AlphaState>(1);
    store.emplace<BetaState>(2.0);

    store.clear();

    EXPECT_TRUE(store.empty());
    EXPECT_EQ(store.size(), std::size_t { 0 });
    EXPECT_FALSE(store.contains<AlphaState>());
}

TEST(TypeStore, IterationVisitsEveryStoredEntry) {
    TypeStore<State> store {};
    store.emplace<AlphaState>(1);
    store.emplace<BetaState>(2.0);

    // Order is unspecified (hash order); assert only that every entry is visited.
    std::size_t count = 0;
    for (const auto& entry : store) {
        EXPECT_NE(entry.second, nullptr);
        ++count;
    }
    EXPECT_EQ(count, std::size_t { 2 });
}

TEST(TypeStore, IsMoveOnly) {
    // unique_ptr elements make the store move-only; copying must be deleted.
    static_assert(!std::is_copy_constructible_v<TypeStore<State>>);
    static_assert(!std::is_copy_assignable_v<TypeStore<State>>);
    static_assert(std::is_move_constructible_v<TypeStore<State>>);
    static_assert(std::is_move_assignable_v<TypeStore<State>>);

    EXPECT_FALSE(std::is_copy_constructible_v<TypeStore<State>>);
}

TEST(TypeStore, MoveConstructionTransfersOwnership) {
    TypeStore<State> source {};
    source.emplace<AlphaState>(5);

    const TypeStore<State> moved = std::move(source);

    ASSERT_NE(moved.get<AlphaState>(), nullptr);
    EXPECT_EQ(moved.get<AlphaState>()->tag, 5);
    EXPECT_TRUE(source.empty());
}

TEST(TypeStore, MoveAssignmentTransfersOwnership) {
    TypeStore<State> source {};
    source.emplace<BetaState>(4.0);

    TypeStore<State> target {};
    target.emplace<AlphaState>(1);
    target = std::move(source);

    // The prior contents of target are dropped; only source's entry remains.
    EXPECT_EQ(target.size(), std::size_t { 1 });
    ASSERT_NE(target.get<BetaState>(), nullptr);
    EXPECT_DOUBLE_EQ(target.get<BetaState>()->value, 4.0);
    EXPECT_EQ(target.get<AlphaState>(), nullptr);
}
