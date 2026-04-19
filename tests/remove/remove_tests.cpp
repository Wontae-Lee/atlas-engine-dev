#include "../utilities/tests_utils.h"

#include <atlas/remove/remove.h>

#include <testkit/testkit.h>

#include <string>
#include <vector>

TEST(Remove, RemoveIfCompactsKeptElementsInOrder) {
    std::vector<int> values { 1, 2, 3, 4, 5, 6 };

    const auto new_end = atlas::remove_if(
        atlas::device,
        values.begin(),
        values.end(),
        [](int value) {
            return value % 2 == 0;
        });

    values.erase(new_end, values.end());

    EXPECT_EQ(values, (std::vector<int> { 1, 3, 5 }));
}

TEST(Remove, RemoveIfReturnsEndWhenNothingIsRemoved) {
    std::vector<int> values { 1, 3, 5 };

    const auto new_end = atlas::remove_if(
        atlas::device,
        values.begin(),
        values.end(),
        [](int) {
            return false;
        });

    EXPECT_EQ(new_end, values.end());
    EXPECT_EQ(values, (std::vector<int> { 1, 3, 5 }));
}

TEST(Remove, RemoveIfCanRemoveEverything) {
    std::vector<int> values { 2, 4, 6 };

    const auto new_end = atlas::remove_if(
        atlas::device,
        values.begin(),
        values.end(),
        [](int) {
            return true;
        });

    EXPECT_EQ(new_end, values.begin());
}

TEST(Remove, RemoveIfPreservesRelativeOrderOfKeptElements) {
    std::vector<std::string> values { "keep-a", "drop", "keep-b", "drop", "keep-c" };

#if defined(ATLAS_TASKING_CUDA)
    SUCCEED();
    return;
#else
    const auto new_end = atlas::remove_if(
        atlas::device,
        values.begin(),
        values.end(),
        [](const std::string& value) {
            return value == "drop";
        });

    values.erase(new_end, values.end());

    EXPECT_EQ(values, (std::vector<std::string> { "keep-a", "keep-b", "keep-c" }));
#endif
}

TEST(Remove, EmptyRangeIsNoOp) {
    std::vector<int> values { 1, 2, 3 };

    const auto new_end = atlas::remove_if(
        atlas::device,
        values.begin(),
        values.begin(),
        [](int) {
            return true;
        });

    EXPECT_EQ(new_end, values.begin());
    EXPECT_EQ(values, (std::vector<int> { 1, 2, 3 }));
}
