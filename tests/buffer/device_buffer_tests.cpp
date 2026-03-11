#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(DeviceBuffer, DefaultConstructedIsEmpty) {

    // A default-constructed DeviceBuffer must represent an empty container.
    //
    // Contract we verify here:
    //  - size() must return 0
    //  - no elements are logically stored
    //
    // The actual device allocation state (null pointer vs reserved memory)
    // is an implementation detail and therefore not checked in this test.
    const DeviceBuffer<int> buf;

    // Explicit cast avoids signed/unsigned comparison warnings.
    EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));
}

TEST(DeviceBuffer, ResizeChangesSize) {

    // Start from an empty buffer.
    DeviceBuffer<int> buf;

    // Growing the buffer must update the logical element count.
    // This mirrors std::vector semantics.
    buf.resize(5);

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(5));

    // Shrinking the buffer must also update the logical element count.
    // The prefix elements may remain in memory internally,
    // but logically only the first `size()` elements are valid.
    buf.resize(2);

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(2));
}

TEST(DeviceBuffer, WriteAndReadElements) {

    // Allocate a small buffer and verify indexing access works.
    //
    // Assumption:
    // DeviceBuffer<T>::operator[] is host-accessible in this build.
    // This may be implemented through:
    //  - unified memory
    //  - mapped memory
    //  - a debug mirror buffer
    //
    // If operator[] becomes device-only in the future,
    // this test must instead use explicit copy APIs or kernels.
    DeviceBuffer<int> buf;
    buf.resize(4);

    // Write a deterministic pattern into the buffer.
    // This verifies both indexing correctness and storage semantics.
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<int>(i * 10);
    }

    // Validate values were written correctly.
    EXPECT_EQ(buf[0], 0);
    EXPECT_EQ(buf[1], 10);
    EXPECT_EQ(buf[2], 20);
    EXPECT_EQ(buf[3], 30);
}

TEST(DeviceBuffer, CopyConstructorPreservesData) {

    // Create a source buffer with known values.
    DeviceBuffer<int> a;
    a.resize(3);

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    // Copy construct a new buffer.
    //
    // Expected semantics:
    //  - size is preserved
    //  - element values are preserved
    //  - device memory must be independent (deep copy)
    const DeviceBuffer<int> b(a);

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));

    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
}

TEST(DeviceBuffer, CopyAssignmentPreservesData) {

    // Create a source buffer with known values.
    DeviceBuffer<int> a;
    a.resize(3);

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    // Copy initialization / assignment.
    //
    // Regardless of whether the implementation uses copy constructor
    // or copy assignment internally, the resulting buffer must contain
    // the same logical contents as the source.
    const DeviceBuffer<int> b = a;

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));

    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
}

TEST(DeviceBuffer, SelfAssignmentIsNoOp) {

    // Prepare a buffer with known values.
    DeviceBuffer<int> a;
    a.resize(3);

    a[0] = 4;
    a[1] = 5;
    a[2] = 6;

    // Self-assignment must be safe.
    //
    // A robust copy/move assignment operator must detect or tolerate
    // the case where the source and destination are identical.
    //
    // Incorrect implementations may:
    //  - free memory prematurely
    //  - corrupt the buffer
    //  - produce undefined behavior
    // ReSharper disable once CppIdenticalOperandsInBinaryExpression
    a = a;

    ASSERT_EQ(a.size(), static_cast<std::size_t>(3));

    EXPECT_EQ(a[0], 4);
    EXPECT_EQ(a[1], 5);
    EXPECT_EQ(a[2], 6);
}

TEST(DeviceBuffer, ResizeKeepsPrefixValuesWhenGrowing) {

    // Initialize a buffer with a known prefix.
    DeviceBuffer<int> buf;
    buf.resize(3);

    buf[0] = 7;
    buf[1] = 8;
    buf[2] = 9;

    // Grow the buffer.
    //
    // Expected semantics (vector-like behavior):
    //  - existing elements [0..old_size-1] must remain unchanged
    //  - newly created elements [old_size..new_size-1] are unspecified
    buf.resize(5);

    ASSERT_EQ(buf.size(), static_cast<std::size_t>(5));

    EXPECT_EQ(buf[0], 7);
    EXPECT_EQ(buf[1], 8);
    EXPECT_EQ(buf[2], 9);

    // buf[3] and buf[4] are intentionally not checked because
    // their initialization behavior is not specified.
}

TEST(DeviceBuffer, ResizeKeepsPrefixValuesWhenShrinking) {

    // Initialize a larger buffer.
    DeviceBuffer<int> buf;
    buf.resize(5);

    buf[0] = 10;
    buf[1] = 11;
    buf[2] = 12;
    buf[3] = 13;
    buf[4] = 14;

    // Shrink the buffer.
    //
    // Expected semantics:
    //  - logical size becomes smaller
    //  - prefix elements remain unchanged
    buf.resize(3);

    ASSERT_EQ(buf.size(), static_cast<std::size_t>(3));

    EXPECT_EQ(buf[0], 10);
    EXPECT_EQ(buf[1], 11);
    EXPECT_EQ(buf[2], 12);

    // Access beyond index 2 would now be undefined behavior
    // and therefore is intentionally not tested.
}

TEST(DeviceBuffer, MoveConstructorTransfersOwnership) {

    // Prepare a source buffer.
    DeviceBuffer<int> a;
    a.resize(3);

    a[0] = 21;
    a[1] = 22;
    a[2] = 23;

    // Move construction transfers ownership of device memory.
    //
    // Expected semantics:
    //  - b receives the device allocation
    //  - a becomes a valid but empty container
    const DeviceBuffer<int> b(std::move(a));

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));

    EXPECT_EQ(b[0], 21);
    EXPECT_EQ(b[1], 22);
    EXPECT_EQ(b[2], 23);

    // Project convention: moved-from buffer reports size 0.
    EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

TEST(DeviceBuffer, MoveAssignmentTransfersOwnership) {

    // Prepare a source buffer.
    DeviceBuffer<int> a;
    a.resize(3);

    a[0] = 31;
    a[1] = 32;
    a[2] = 33;

    // Prepare a destination buffer with different content.
    DeviceBuffer<int> b;
    b.resize(2);

    b[0] = -1;
    b[1] = -2;

    // Move assignment.
    //
    // Expected semantics:
    //  - destination releases its previous allocation
    //  - destination takes ownership of source allocation
    //  - source becomes empty but valid
    b = std::move(a);

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));

    EXPECT_EQ(b[0], 31);
    EXPECT_EQ(b[1], 32);
    EXPECT_EQ(b[2], 33);

    EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

TEST(DeviceBuffer, ResizeToZeroClearsSize) {

    // Create a buffer with several elements.
    DeviceBuffer<int> buf;
    buf.resize(4);

    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    buf[3] = 4;

    // Resize to zero.
    //
    // Expected semantics:
    //  - logical size becomes 0
    //
    // Whether device memory is freed or kept for reuse
    // is an implementation detail and therefore not verified here.
    buf.resize(0);

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));
}