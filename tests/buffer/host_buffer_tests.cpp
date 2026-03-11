#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(HostBuffer, DefaultConstructedIsEmpty) {

    // Default-constructed HostBuffer must represent an empty container:
    // - size() must be 0
    // - underlying device allocation is expected to be null/empty (implementation detail)
    const HostBuffer<int> buf;

    // Use explicit std::size_t to avoid signed/unsigned warnings.
    EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));

    // Log the size for debugging purposes (optional).
    logger::info() << "\n"
                   << "Default-constructed HostBuffer has size: "
                   << buf.size()
                   << " (expected 0)";
}

TEST(HostBuffer, ResizeChangesSize) {

    // Start from an empty buffer and verify resize updates the logical element count.
    HostBuffer<int> buf;

    // Growing resize must set size() to the requested number of elements.
    buf.resize(5);

    // Log the size after resizing for debugging purposes (optional).
    logger::info() << "\n"
                   << "After resizing to 5, HostBuffer size is: "
                   << buf.size()
                   << " (expected 5)";

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(5));

    // Shrinking resize must reduce size() accordingly.
    buf.resize(2);

    // Log the size after shrinking for debugging purposes (optional).
    logger::info() << "\n"
                   << "After resizing to 2, HostBuffer size is: "
                   << buf.size()
                   << " (expected 2)";
    EXPECT_EQ(buf.size(), static_cast<std::size_t>(2));
}

TEST(HostBuffer, WriteAndReadElements) {

    // Allocate a small device buffer and verify operator[] can write/read elements.
    //
    // Note:
    // - This assumes HostBuffer<T>::operator[] is host-accessible in this project
    //   (e.g., via unified memory, mapped memory, or a debug host mirror).
    // - If operator[] is device-only in some configurations, this test will need
    //   to be adapted to use explicit copy APIs or kernels.
    HostBuffer<int> buf;
    buf.resize(4);

    // Write a known pattern to validate indexing and storage semantics.
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<int>(i * 10);
    }

    // Log the buffer contents for debugging purposes (optional).
    logger::info() << "\n"
                   << "Buffer contents after writing:\n"
                   << "buf[0] = " << buf[0] << "\n"
                   << "buf[1] = " << buf[1] << "\n"
                   << "buf[2] = " << buf[2] << "\n"
                   << "buf[3] = " << buf[3] << "\n";

    // Read back and verify values were stored correctly.
    EXPECT_EQ(buf[0], 0);
    EXPECT_EQ(buf[1], 10);
    EXPECT_EQ(buf[2], 20);
    EXPECT_EQ(buf[3], 30);
}

TEST(HostBuffer, CopyConstructorPreservesData) {

    // Prepare an input buffer with known contents.
    HostBuffer<int> a;
    a.resize(3);
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    // Copy-construct a new buffer.
    // Expected semantics:
    // - `b` has the same size as `a`
    // - element values are preserved
    // - storage is independent (deep copy) in device memory (implementation detail)
    const HostBuffer<int> b(a);

    // Log the copied buffer contents for debugging purposes (optional).
    logger::info() << "\n"
                   << "Copied buffer contents:\n"
                   << "b[0] = " << b[0] << "\n"
                   << "b[1] = " << b[1] << "\n"
                   << "b[2] = " << b[2] << "\n";

    // Validate size and values were copied.
    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
}

TEST(HostBuffer, CopyAssignmentPreservesData) {

    // Prepare an input buffer with known contents.
    HostBuffer<int> a;
    a.resize(3);
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    // Copy-initialize/assign a second buffer from `a`.
    // This exercises copy construction or copy assignment depending on the API,
    // but in either case the resulting buffer must contain the same data.
    const HostBuffer<int> b = a;

    // Log the copied buffer contents for debugging purposes (optional).
    logger::info() << "\n"
                   << "Copy-assigned buffer contents:\n"
                   << "b[0] = " << b[0] << "\n"
                   << "b[1] = " << b[1] << "\n"
                   << "b[2] = " << b[2] << "\n";

    // Validate size and values were copied.
    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
}

TEST(HostBuffer, SelfAssignmentIsNoOp) {

    // Prepare a buffer with known content.
    HostBuffer<int> a;
    a.resize(3);
    a[0] = 4;
    a[1] = 5;
    a[2] = 6;

    // Self-assignment should be safe and should not corrupt or free the buffer.
    // This is a common edge case for move/copy assignment implementations.
    // ReSharper disable once CppIdenticalOperandsInBinaryExpression
    a = a;
    // Log the buffer contents after self-assignment for debugging purposes (optional).
    logger::info() << "\n"
                   << "Buffer contents after self-assignment:\n"
                   << "a[0] = " << a[0] << "\n"
                   << "a[1] = " << a[1] << "\n"
                   << "a[2] = " << a[2] << "\n";

    // Validate size and values are unchanged.
    ASSERT_EQ(a.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(a[0], 4);
    EXPECT_EQ(a[1], 5);
    EXPECT_EQ(a[2], 6);
}

TEST(HostBuffer, ResizeKeepsPrefixValuesWhenGrowing) {

    // Start with a small buffer and initialize a known prefix.
    HostBuffer<int> buf;
    buf.resize(3);
    buf[0] = 7;
    buf[1] = 8;
    buf[2] = 9;

    // Grow the buffer.
    // Expected semantics (vector-like):
    // - old elements [0..old_size-1] are preserved
    // - new elements [old_size..new_size-1] are unspecified unless documented otherwise
    buf.resize(5);

    // Log the buffer contents after resizing for debugging purposes (optional).
    logger::info() << "\n"
                   << "Buffer contents after resizing to larger size:\n"
                   << "buf[0] = " << buf[0] << "\n"
                   << "buf[1] = " << buf[1] << "\n"
                   << "buf[2] = " << buf[2] << "\n"
                   << "buf[3] = " << buf[3] << " (unspecified)\n"
                   << "buf[4] = " << buf[4] << " (unspecified)\n";

    // Validate size and preserved prefix.
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
    // Expected semantics:
    // - size is reduced
    // - elements in the surviving prefix remain unchanged
    buf.resize(3);

    // Log the buffer contents after shrinking for debugging purposes (optional).
    logger::info() << "\n"
                   << "Buffer contents after resizing to smaller size:\n"
                   << "buf[0] = " << buf[0] << "\n"
                   << "buf[1] = " << buf[1] << "\n"
                   << "buf[2] = " << buf[2] << "\n"
                   << "buf[3] = (out of bounds)\n"
                   << "buf[4] = (out of bounds)\n";

    // Validate size and preserved prefix.
    ASSERT_EQ(buf.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(buf[0], 10);
    EXPECT_EQ(buf[1], 11);
    EXPECT_EQ(buf[2], 12);
}

TEST(HostBuffer, MoveConstructorTransfersOwnership) {

    // Prepare a source buffer with known content.
    HostBuffer<int> a;
    a.resize(3);
    a[0] = 21;
    a[1] = 22;
    a[2] = 23;

    // Move-construct a new buffer from `a`.
    // Expected semantics:
    // - `b` takes ownership of `a`'s device allocation
    // - `a` becomes empty/valid (project convention: size() == 0)
    const HostBuffer<int> b(std::move(a));

    // Log the moved buffer contents for debugging purposes (optional).
    logger::info() << "\n"
                   << "Moved buffer contents:\n"
                   << "b[0] = " << b[0] << "\n"
                   << "b[1] = " << b[1] << "\n"
                   << "b[2] = " << b[2] << "\n"
                   << "a.size() = " << a.size() << " (expected 0)\n";

    // Validate destination has the moved data.
    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 21);
    EXPECT_EQ(b[1], 22);
    EXPECT_EQ(b[2], 23);

    // Validate moved-from state.
    EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

TEST(HostBuffer, MoveAssignmentTransfersOwnership) {

    // Prepare a source buffer with known content.
    HostBuffer<int> a;
    a.resize(3);
    a[0] = 31;
    a[1] = 32;
    a[2] = 33;

    // Prepare a destination buffer with different content to ensure it gets replaced.
    HostBuffer<int> b;
    b.resize(2);
    b[0] = -1;
    b[1] = -2;

    // Move-assign `a` into `b`.
    // Expected semantics:
    // - `b` releases its previous allocation (if any)
    // - `b` takes ownership of `a`'s allocation
    // - `a` becomes empty/valid (project convention)
    b = std::move(a);

    // Log the moved buffer contents for debugging purposes (optional).
    logger::info() << "\n"
                   << "Moved buffer contents after move assignment:\n"
                   << "b[0] = " << b[0] << "\n"
                   << "b[1] = " << b[1] << "\n"
                   << "b[2] = " << b[2] << "\n"
                   << "a.size() = " << a.size() << " (expected 0)\n";

    // Validate destination now holds the moved data.
    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 31);
    EXPECT_EQ(b[1], 32);
    EXPECT_EQ(b[2], 33);

    // Validate moved-from state.
    EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

TEST(HostBuffer, ResizeToZeroClearsSize) {

    // Prepare a buffer with some content.
    HostBuffer<int> buf;
    buf.resize(4);
    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    buf[3] = 4;

    // Resize to zero should clear the logical size.
    // Whether the underlying device memory is freed is an implementation detail.
    buf.resize(0);

    // Log the buffer size after resizing to zero for debugging purposes (optional).
    logger::info() << "\n"
                   << "Buffer size after resizing to zero: "
                   << buf.size()
                   << " (expected 0)";

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));
}
