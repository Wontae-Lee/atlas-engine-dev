#include <atlas/buffer/device_buffer.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace {

using atlas::DeviceBuffer;

// A DeviceBuffer element must never be dereferenced in place: under the CUDA backend
// operator[] yields a device_reference proxy, so every read copies into a host value
// first. This mirrors the round-trip pattern in material_dictionary_tests.cpp.

}

TEST(DeviceBuffer, DefaultConstructionIsEmpty) {
    const DeviceBuffer<int> buffer {};

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), std::size_t { 0 });
}

TEST(DeviceBuffer, SizedConstructionValueInitializes) {
    const DeviceBuffer<int> buffer(4);

    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size(), std::size_t { 4 });

    for (std::size_t i = 0; i < buffer.size(); ++i) {
        const int element = buffer[i];
        EXPECT_EQ(element, 0);
    }
}

TEST(DeviceBuffer, SizedWithValueConstructionFillsEveryElement) {
    const DeviceBuffer<int> buffer(3, 7);

    EXPECT_EQ(buffer.size(), std::size_t { 3 });

    for (std::size_t i = 0; i < buffer.size(); ++i) {
        const int element = buffer[i];
        EXPECT_EQ(element, 7);
    }
}

TEST(DeviceBuffer, IteratorRangeConstructionCopiesSource) {
    const std::vector<int> source { 10, 20, 30, 40 };

    const DeviceBuffer<int> buffer(source.begin(), source.end());

    ASSERT_EQ(buffer.size(), source.size());
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        const int element = buffer[i];
        EXPECT_EQ(element, source[i]);
    }
}

TEST(DeviceBuffer, ResizePreservesPrefixContents) {
    const std::vector<int> source { 1, 2, 3 };
    DeviceBuffer<int> buffer(source.begin(), source.end());

    buffer.resize(5);

    ASSERT_EQ(buffer.size(), std::size_t { 5 });
    for (std::size_t i = 0; i < source.size(); ++i) {
        const int element = buffer[i];
        EXPECT_EQ(element, source[i]);
    }
}

TEST(DeviceBuffer, ResizeShrinkKeepsRemainingPrefix) {
    const std::vector<int> source { 5, 6, 7, 8 };
    DeviceBuffer<int> buffer(source.begin(), source.end());

    buffer.resize(2);

    ASSERT_EQ(buffer.size(), std::size_t { 2 });
    const int first  = buffer[0];
    const int second = buffer[1];
    EXPECT_EQ(first, 5);
    EXPECT_EQ(second, 6);
}
