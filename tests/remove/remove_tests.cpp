#include <atlas/remove/remove.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/copy.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using atlas::ExecutionPolicy;
using atlas::remove_if;

struct IsEven {
    ATLAS_ALL_DEVICE bool
    operator()(const int value) const {
        return value % 2 == 0;
    }
};

}

TEST(Remove, RemoveIfCompactsKeptElementsInOrder) {
    std::vector<int> values { 1, 2, 3, 4, 5, 6 };

    const auto new_end = remove_if<ExecutionPolicy::host>(
        values.begin(),
        values.end(),
        [](int value) {
            return value % 2 == 0;
        });

    values.erase(new_end, values.end());

    EXPECT_EQ(values, (std::vector<int> { 1, 3, 5 }));
}

TEST(Remove, DeviceRemoveIfCompactsKeptElementsInOrder) {
    const std::vector<int> input { 1, 2, 3, 4, 5, 6 };
    atlas::DeviceBuffer<int> values(input.size());
    atlas::copy_host_to_device(input.data(), values, input.size());

    const auto new_end = remove_if<ExecutionPolicy::device>(
        values.begin(),
        values.end(),
        IsEven {});

    const auto kept = static_cast<std::size_t>(new_end - values.begin());
    ASSERT_EQ(kept, 3u);

    std::vector<int> host_values(kept, 0);
    atlas::copy_device_to_host(values, host_values.data(), kept);

    EXPECT_EQ(host_values, (std::vector<int> { 1, 3, 5 }));
}

TEST(Remove, RemoveIfReturnsEndWhenNothingIsRemoved) {
    std::vector<int> values { 1, 3, 5 };

    const auto new_end = remove_if<ExecutionPolicy::host>(
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

    const auto new_end = remove_if<ExecutionPolicy::host>(
        values.begin(),
        values.end(),
        [](int) {
            return true;
        });

    EXPECT_EQ(new_end, values.begin());
}

TEST(Remove, RemoveIfPreservesRelativeOrderOfKeptElements) {
    std::vector<std::string> values { "keep-a", "drop", "keep-b", "drop", "keep-c" };

    const auto new_end = remove_if<ExecutionPolicy::host>(
        values.begin(),
        values.end(),
        [](const std::string& value) {
            return value == "drop";
        });

    values.erase(new_end, values.end());

    EXPECT_EQ(values, (std::vector<std::string> { "keep-a", "keep-b", "keep-c" }));
}

TEST(Remove, EmptyRangeIsNoOp) {
    std::vector<int> values { 1, 2, 3 };

    const auto new_end = remove_if<ExecutionPolicy::host>(
        values.begin(),
        values.begin(),
        [](int) {
            return true;
        });

    EXPECT_EQ(new_end, values.begin());
    EXPECT_EQ(values, (std::vector<int> { 1, 2, 3 }));
}
