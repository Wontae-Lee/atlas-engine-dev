#include "../utilities/tests_utils.h"

#include <atlas/buffer/host_buffer.h>
#include <testkit/testkit.h>

using namespace atlas;

TEST(HostBuffer, DefaultConstructedIsEmpty) {

    const HostBuffer<int> buf;

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));
}

TEST(HostBuffer, ResizeChangesSize) {

    HostBuffer<int> buf;

    buf.resize(5);

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(5));

    buf.resize(2);

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(2));
}

TEST(HostBuffer, WriteAndReadElements) {

    HostBuffer<int> buf;
    buf.resize(4);

    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<int>(i * 10);
    }

    EXPECT_EQ(buf[0], 0);
    EXPECT_EQ(buf[1], 10);
    EXPECT_EQ(buf[2], 20);
    EXPECT_EQ(buf[3], 30);
}

TEST(HostBuffer, CopyConstructorPreservesData) {

    HostBuffer<int> a;
    a.resize(3);

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    const HostBuffer<int> b(a);

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
}

TEST(HostBuffer, CopyAssignmentPreservesData) {

    HostBuffer<int> a;
    a.resize(3);

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    const HostBuffer<int> b = a;

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
}

TEST(HostBuffer, SelfAssignmentIsNoOp) {

    HostBuffer<int> a;
    a.resize(3);

    a[0] = 4;
    a[1] = 5;
    a[2] = 6;

    a = a;

    ASSERT_EQ(a.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(a[0], 4);
    EXPECT_EQ(a[1], 5);
    EXPECT_EQ(a[2], 6);
}

TEST(HostBuffer, ResizeKeepsPrefixValuesWhenGrowing) {

    HostBuffer<int> buf;
    buf.resize(3);

    buf[0] = 7;
    buf[1] = 8;
    buf[2] = 9;

    buf.resize(5);

    ASSERT_EQ(buf.size(), static_cast<std::size_t>(5));
    EXPECT_EQ(buf[0], 7);
    EXPECT_EQ(buf[1], 8);
    EXPECT_EQ(buf[2], 9);
}

TEST(HostBuffer, ResizeKeepsPrefixValuesWhenShrinking) {

    HostBuffer<int> buf;
    buf.resize(5);

    buf[0] = 10;
    buf[1] = 11;
    buf[2] = 12;
    buf[3] = 13;
    buf[4] = 14;

    buf.resize(3);

    ASSERT_EQ(buf.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(buf[0], 10);
    EXPECT_EQ(buf[1], 11);
    EXPECT_EQ(buf[2], 12);
}

TEST(HostBuffer, MoveConstructorTransfersOwnership) {

    HostBuffer<int> a;
    a.resize(3);

    a[0] = 21;
    a[1] = 22;
    a[2] = 23;

    const HostBuffer<int> b(std::move(a));

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 21);
    EXPECT_EQ(b[1], 22);
    EXPECT_EQ(b[2], 23);

    EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

TEST(HostBuffer, MoveAssignmentTransfersOwnership) {

    HostBuffer<int> a;
    a.resize(3);

    a[0] = 31;
    a[1] = 32;
    a[2] = 33;

    HostBuffer<int> b;
    b.resize(2);

    b[0] = -1;
    b[1] = -2;

    b = std::move(a);

    ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(b[0], 31);
    EXPECT_EQ(b[1], 32);
    EXPECT_EQ(b[2], 33);

    EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

TEST(HostBuffer, ResizeToZeroClearsSize) {

    HostBuffer<int> buf;
    buf.resize(4);

    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    buf[3] = 4;

    buf.resize(0);

    EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));
}