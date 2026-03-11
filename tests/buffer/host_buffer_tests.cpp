#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(HostBuffer, DefaultConstructedIsEmpty) {

    // A default-constructed HostBuffer must behave as an empty container.
    //
    // Contract verified here:
    //  - size() returns 0
    //  - no elements are logically available
    //
    // The exact internal allocation strategy is not checked because that is
    // an implementation detail rather than part of the observable API.
    const HostBuffer<int> buf;

    // Use an explicit std::size_t value to keep the comparison type-safe.
    EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));
}

TEST(HostBuffer, ResizeChangesSize) {

    // Begin with an empty buffer.
    HostBuffer<int> buf;

    // Growing the buffer must update the logical number of stored elements.
    // This test checks only the public size contract, not the underlying
    // allocation behavior.
    buf.resize(5);

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(5));

    // Shrinking must also update the logical size accordingly.
    // Elements beyond the new size are no longer part of the valid range.
    buf.resize(2);

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(2));
}

TEST(HostBuffer, WriteAndReadElements) {

    // Allocate a small host buffer and verify indexed write/read access.
    //
    // Because this is HostBuffer, operator[] is expected to be directly
    // accessible on the host side. This test validates:
    //  - indexing into valid positions
    //  - writing values
    //  - reading the same values back
    HostBuffer<int> buf;
    buf.resize(4);

    // Write a deterministic pattern so every index has a distinct value.
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<int>(i * 10);
    }

    // Read back the values and verify the stored contents.
    EXPECT_EQ(buf[0], 0);
    EXPECT_EQ(buf[1], 10);
    EXPECT_EQ(buf[2], 20);
    EXPECT_EQ(buf[3], 30);
}

TEST(HostBuffer, CopyConstructorPreservesData) {

    // Prepare a source buffer with known contents.
    HostBuffer<int> a;
    a.resize(3);

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    // Copy construct a second buffer.
    //
    // Expected semantics:
    //  - copied buffer has the same size
    //  - copied buffer contains the same values
    //  - storage is independent (deep-copy behavior)
    const HostBuffer<int> b(a);

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
}

TEST(HostBuffer, CopyAssignmentPreservesData) {

    // Prepare a source buffer with known contents.
    HostBuffer<int> a;
    a.resize(3);

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    // Initialize another buffer from the source.
    //
    // This verifies that the destination receives an equivalent logical state:
    // same size and same values.
    const HostBuffer<int> b = a;

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
}

TEST(HostBuffer, SelfAssignmentIsNoOp) {

    // Prepare a buffer with known contents.
    HostBuffer<int> a;
    a.resize(3);

    a[0] = 4;
    a[1] = 5;
    a[2] = 6;

    // ReSharper disable once CppIdenticalOperandsInBinaryExpression
    a = a;

    ASSERT_EQ(a.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(a[0], 4);
    EXPECT_EQ(a[1], 5);
    EXPECT_EQ(a[2], 6);
}

TEST(HostBuffer, ResizeKeepsPrefixValuesWhenGrowing) {

    // Start with a small buffer and initialize all current elements.
    HostBuffer<int> buf;
    buf.resize(3);

    buf[0] = 7;
    buf[1] = 8;
    buf[2] = 9;

    // Grow the buffer.
    //
    // Expected vector-like semantics:
    //  - the old prefix remains unchanged
    //  - newly added elements may be default-initialized or unspecified,
    //    depending on implementation, so they are not checked here
    buf.resize(5);

    ASSERT_EQ(buf.size(), static_cast<std::size_t>(5));
    EXPECT_EQ(buf[0], 7);
    EXPECT_EQ(buf[1], 8);
    EXPECT_EQ(buf[2], 9);
}

TEST(HostBuffer, ResizeKeepsPrefixValuesWhenShrinking) {

    // Start with a larger buffer and initialize all elements.
    HostBuffer<int> buf;
    buf.resize(5);

    buf[0] = 10;
    buf[1] = 11;
    buf[2] = 12;
    buf[3] = 13;
    buf[4] = 14;

    // Shrink the buffer.
    //
    // Expected semantics:
    //  - size is reduced
    //  - values in the surviving prefix remain unchanged
    //  - removed elements are no longer accessible and must not be read
    buf.resize(3);

    ASSERT_EQ(buf.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(buf[0], 10);
    EXPECT_EQ(buf[1], 11);
    EXPECT_EQ(buf[2], 12);
}

TEST(HostBuffer, MoveConstructorTransfersOwnership) {

    // Prepare a source buffer with known contents.
    HostBuffer<int> a;
    a.resize(3);

    a[0] = 21;
    a[1] = 22;
    a[2] = 23;

    // Move construct a new buffer.
    //
    // Expected semantics:
    //  - destination receives the source contents
    //  - source becomes valid but empty after the move
    const HostBuffer<int> b(std::move(a));

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 21);
    EXPECT_EQ(b[1], 22);
    EXPECT_EQ(b[2], 23);

    // Project convention used in these tests:
    // moved-from buffer reports size 0.
    EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

TEST(HostBuffer, MoveAssignmentTransfersOwnership) {

    // Prepare a source buffer with known contents.
    HostBuffer<int> a;
    a.resize(3);

    a[0] = 31;
    a[1] = 32;
    a[2] = 33;

    // Prepare a destination buffer with different initial contents.
    // This ensures the move assignment replaces the previous state.
    HostBuffer<int> b;
    b.resize(2);

    b[0] = -1;
    b[1] = -2;

    // Move assign from a into b.
    //
    // Expected semantics:
    //  - b takes over the logical contents previously held by a
    //  - prior contents of b are discarded
    //  - a becomes valid but empty
    b = std::move(a);

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 31);
    EXPECT_EQ(b[1], 32);
    EXPECT_EQ(b[2], 33);

    EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

TEST(HostBuffer, ResizeToZeroClearsSize) {

    // Prepare a non-empty buffer.
    HostBuffer<int> buf;
    buf.resize(4);

    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    buf[3] = 4;

    // Resizing to zero must clear the logical contents of the buffer.
    //
    // Whether capacity or underlying allocation is retained for reuse is an
    // implementation detail and not part of this test.
    buf.resize(0);

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));
}