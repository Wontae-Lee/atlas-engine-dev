#include <atlas/scan/exclusive_scan.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace {

using atlas::DeviceBuffer;
using atlas::ExecutionPolicy;

// Copy a device buffer back into a host vector for value comparison.
template <typename T>
std::vector<T>
to_host(const DeviceBuffer<T>& buffer) {
    std::vector<T> host(buffer.size());
    if (!buffer.empty()) {
        atlas::copy_device_to_host(buffer, host.data(), buffer.size());
    }
    return host;
}

// Run an out-of-place exclusive scan on the device backend with an explicit init.
// Mirrors the raw-pointer driving used by the parallel_sort/parallel_fill tests: the
// input is uploaded to a DeviceBuffer, scanned into a second buffer, and read back.
template <typename T>
std::vector<T>
device_scan(const std::vector<T>& input, const T init) {
    DeviceBuffer<T> in(input.begin(), input.end());
    DeviceBuffer<T> out(input.size());

    T* first  = atlas::raw_pointer_cast(in.data());
    T* result = atlas::raw_pointer_cast(out.data());

    atlas::exclusive_scan<ExecutionPolicy::device>(first, first + input.size(), result, init);

    return to_host(out);
}

// Serial reference exclusive scan computed entirely on the host.
template <typename T>
std::vector<T>
serial_reference(const std::vector<T>& input, const T init) {
    std::vector<T> reference(input.size());
    std::exclusive_scan(input.begin(), input.end(), reference.begin(), init);
    return reference;
}

// A host+device associative "running maximum" combiner. Annotated so the CUDA device
// branch of exclusive_scan can instantiate it, unlike a bare std::max functor.
struct MaxOp {
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
    operator()(const int a, const int b) const noexcept {
        return a > b ? a : b;
    }
};

}

TEST(ExclusiveScan, EmptyInputWritesNothingAndReturnsResult) {
    DeviceBuffer<int> in(std::size_t { 0 });
    DeviceBuffer<int> out(std::size_t { 0 });

    int* first  = atlas::raw_pointer_cast(in.data());
    int* result = atlas::raw_pointer_cast(out.data());

    int* returned = atlas::exclusive_scan<ExecutionPolicy::device>(first, first, result);

    // An empty range performs no writes and returns the result iterator unchanged.
    EXPECT_EQ(returned, result);
}

TEST(ExclusiveScan, SingleElementYieldsInit) {
    // With one input the only output slot is the init value; the input itself is
    // never folded in (it has no predecessor prefix).
    const std::vector<int> result = device_scan(std::vector<int> { 99 }, 0);

    ASSERT_EQ(result.size(), std::size_t { 1 });
    EXPECT_EQ(result[0], 0);
}

TEST(ExclusiveScan, PrefixSumMatchesCanonicalResult) {
    const std::vector<int> input { 3, 1, 4, 1, 5 };

    const std::vector<int> result = device_scan(input, 0);

    // result[0] is the identity, result[i] is the sum of everything before i.
    EXPECT_EQ(result, (std::vector<int> { 0, 3, 4, 8, 9 }));

    // The first output is the init value; the last is the sum of all but the last input.
    EXPECT_EQ(result.front(), 0);
    EXPECT_EQ(result.back(), 3 + 1 + 4 + 1);
}

TEST(ExclusiveScan, NonZeroInitOffsetsEveryPrefix) {
    const std::vector<int> input { 3, 1, 4, 1, 5 };

    const std::vector<int> result = device_scan(input, 10);

    // Each prefix is the zero-init scan plus the init value.
    EXPECT_EQ(result, (std::vector<int> { 10, 13, 14, 18, 19 }));
}

TEST(ExclusiveScan, DefaultOverloadUsesZeroInit) {
    // The two-argument overload value-initializes the init (0 for int).
    const std::vector<int> input { 2, 7, 1, 8 };

    DeviceBuffer<int> in(input.begin(), input.end());
    DeviceBuffer<int> out(input.size());

    int* first  = atlas::raw_pointer_cast(in.data());
    int* result = atlas::raw_pointer_cast(out.data());

    atlas::exclusive_scan<ExecutionPolicy::device>(first, first + input.size(), result);

    EXPECT_EQ(to_host(out), (std::vector<int> { 0, 2, 9, 10 }));
}

TEST(ExclusiveScan, CustomBinaryOpComputesRunningMax) {
    const std::vector<int> input { 3, 1, 4, 1, 5 };

    DeviceBuffer<int> in(input.begin(), input.end());
    DeviceBuffer<int> out(input.size());

    int* first  = atlas::raw_pointer_cast(in.data());
    int* result = atlas::raw_pointer_cast(out.data());

    atlas::exclusive_scan<ExecutionPolicy::device>(first, first + input.size(), result, 0, MaxOp {});

    // Exclusive running maximum: out[i] = max(init, in[0..i-1]).
    EXPECT_EQ(to_host(out), (std::vector<int> { 0, 3, 3, 4, 4 }));
}

TEST(ExclusiveScan, LargeInputMatchesSerialReference) {
    // A 10k-element scan checked against a host std::exclusive_scan reference. The
    // values stay small so the running int sum never overflows.
    const std::size_t count = 10000;
    std::vector<int>  input(count);
    for (std::size_t i = 0; i < count; ++i) {
        input[i] = static_cast<int>(i % 7);
    }

    EXPECT_EQ(device_scan(input, 0), serial_reference(input, 0));
}

TEST(ExclusiveScan, InPlaceScanMatchesOutOfPlace) {
    const std::vector<int> input { 3, 1, 4, 1, 5, 9, 2, 6 };

    // The result iterator aliases the input for an in-place scan.
    DeviceBuffer<int> buffer(input.begin(), input.end());
    int*              first = atlas::raw_pointer_cast(buffer.data());

    atlas::exclusive_scan<ExecutionPolicy::device>(first, first + input.size(), first, 0);

    EXPECT_EQ(to_host(buffer), serial_reference(input, 0));
}

TEST(ExclusiveScan, ReturnsIteratorPastLastWritten) {
    const std::vector<int> input { 1, 2, 3, 4 };

    DeviceBuffer<int> in(input.begin(), input.end());
    DeviceBuffer<int> out(input.size());

    int* first  = atlas::raw_pointer_cast(in.data());
    int* result = atlas::raw_pointer_cast(out.data());

    int* returned = atlas::exclusive_scan<ExecutionPolicy::device>(first, first + input.size(), result, 0);

    // The return is the output iterator advanced by the input length.
    EXPECT_EQ(returned, result + input.size());
}

TEST(ExclusiveScan, UnsignedElementType) {
    const std::vector<unsigned> input { 5u, 10u, 15u, 20u };

    const std::vector<unsigned> result = device_scan(input, 0u);

    EXPECT_EQ(result, (std::vector<unsigned> { 0u, 5u, 15u, 30u }));
}

TEST(ExclusiveScan, FloatElementType) {
    // Values are exactly representable so the prefix sums are exact regardless of
    // the backend's reduction order.
    const std::vector<float> input { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

    const std::vector<float> result = device_scan(input, 0.5f);

    ASSERT_EQ(result.size(), input.size());
    const std::vector<float> expected { 0.5f, 1.5f, 3.5f, 6.5f, 10.5f };
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}
