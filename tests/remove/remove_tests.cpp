#include "../utilities/test_utils.h"

#include <atlas/remove/remove.h>

#include <testkit/testkit.h>

#include <string>
#include <vector>

namespace {

using atlas::device;
using atlas::remove_if;

} // namespace

TEST(Remove, RemoveIfCompactsKeptElementsInOrder) {
    // Arrange: create a sequence with alternating kept and removed values.
    std::vector<int> values { 1, 2, 3, 4, 5, 6 };

    // Act: remove even values and compact the vector.
    const auto new_end = remove_if(
        device,
        values.begin(),
        values.end(),
        [](int value) {
            return value % 2 == 0;
        });

    values.erase(new_end, values.end());

    // Assert: kept values preserve their relative order.
    EXPECT_EQ(values, (std::vector<int> { 1, 3, 5 }));
}

TEST(Remove, RemoveIfReturnsEndWhenNothingIsRemoved) {
    // Arrange: create a sequence where the predicate never matches.
    std::vector<int> values { 1, 3, 5 };

    // Act: run remove_if with a predicate that keeps every value.
    const auto new_end = remove_if(
        device,
        values.begin(),
        values.end(),
        [](int) {
            return false;
        });

    // Assert: the range end and values are unchanged.
    EXPECT_EQ(new_end, values.end());
    EXPECT_EQ(values, (std::vector<int> { 1, 3, 5 }));
}

TEST(Remove, RemoveIfCanRemoveEverything) {
    // Arrange: create a sequence where every value should be removed.
    std::vector<int> values { 2, 4, 6 };

    // Act: run remove_if with a predicate that removes every value.
    const auto new_end = remove_if(
        device,
        values.begin(),
        values.end(),
        [](int) {
            return true;
        });

    // Assert: the new end is the beginning of the range.
    EXPECT_EQ(new_end, values.begin());
}

TEST(Remove, RemoveIfPreservesRelativeOrderOfKeptElements) {
    // Arrange: create a string sequence with repeated removable markers.
    std::vector<std::string> values { "keep-a", "drop", "keep-b", "drop", "keep-c" };

#if defined(ATLAS_TASKING_CUDA)
    // Assert: string removal is host-only for this test.
    SUCCEED();
    return;
#else
    // Act: remove marker values and compact the vector.
    const auto new_end = remove_if(
        device,
        values.begin(),
        values.end(),
        [](const std::string& value) {
            return value == "drop";
        });

    values.erase(new_end, values.end());

    // Assert: kept strings preserve their relative order.
    EXPECT_EQ(values, (std::vector<std::string> { "keep-a", "keep-b", "keep-c" }));
#endif
}

TEST(Remove, EmptyRangeIsNoOp) {
    // Arrange: create a sequence and choose an empty subrange.
    std::vector<int> values { 1, 2, 3 };

    // Act: run remove_if on an empty range.
    const auto new_end = remove_if(
        device,
        values.begin(),
        values.begin(),
        [](int) {
            return true;
        });

    // Assert: the returned iterator and values are unchanged.
    EXPECT_EQ(new_end, values.begin());
    EXPECT_EQ(values, (std::vector<int> { 1, 2, 3 }));
}
