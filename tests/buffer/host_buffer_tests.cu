#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

using namespace atlas;

CUDA_TEST(HostBuffer, DefaultConstructedIsEmpty) {

    const HostBuffer<int> buf;

    CUDA_EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));
}

CUDA_TEST(HostBuffer, ResizeChangesSize) {

    HostBuffer<int> buf;

    buf.resize(5);

    CUDA_EXPECT_EQ(buf.size(), static_cast<std::size_t>(5));

    buf.resize(2);

    CUDA_EXPECT_EQ(buf.size(), static_cast<std::size_t>(2));
}

CUDA_TEST(HostBuffer, WriteAndReadElements) {

    HostBuffer<int> buf;
    buf.resize(4);

    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<int>(i * 10);
    }

    CUDA_EXPECT_EQ(buf[0], 0);
    CUDA_EXPECT_EQ(buf[1], 10);
    CUDA_EXPECT_EQ(buf[2], 20);
    CUDA_EXPECT_EQ(buf[3], 30);
}

CUDA_TEST(HostBuffer, CopyConstructorPreservesData) {

    HostBuffer<int> a;
    a.resize(3);

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    const HostBuffer<int> b(a);

    CUDA_ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    CUDA_EXPECT_EQ(b[0], 1);
    CUDA_EXPECT_EQ(b[1], 2);
    CUDA_EXPECT_EQ(b[2], 3);
}

CUDA_TEST(HostBuffer, CopyAssignmentPreservesData) {

    HostBuffer<int> a;
    a.resize(3);

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    const HostBuffer<int> b = a;

    CUDA_ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    CUDA_EXPECT_EQ(b[0], 1);
    CUDA_EXPECT_EQ(b[1], 2);
    CUDA_EXPECT_EQ(b[2], 3);
}

CUDA_TEST(HostBuffer, SelfAssignmentIsNoOp) {

    HostBuffer<int> a;
    a.resize(3);

    a[0] = 4;
    a[1] = 5;
    a[2] = 6;

    a = a;

    CUDA_ASSERT_EQ(a.size(), static_cast<std::size_t>(3));
    CUDA_EXPECT_EQ(a[0], 4);
    CUDA_EXPECT_EQ(a[1], 5);
    CUDA_EXPECT_EQ(a[2], 6);
}

CUDA_TEST(HostBuffer, ResizeKeepsPrefixValuesWhenGrowing) {

    HostBuffer<int> buf;
    buf.resize(3);

    buf[0] = 7;
    buf[1] = 8;
    buf[2] = 9;

    buf.resize(5);

    CUDA_ASSERT_EQ(buf.size(), static_cast<std::size_t>(5));
    CUDA_EXPECT_EQ(buf[0], 7);
    CUDA_EXPECT_EQ(buf[1], 8);
    CUDA_EXPECT_EQ(buf[2], 9);
}

CUDA_TEST(HostBuffer, ResizeKeepsPrefixValuesWhenShrinking) {

    HostBuffer<int> buf;
    buf.resize(5);

    buf[0] = 10;
    buf[1] = 11;
    buf[2] = 12;
    buf[3] = 13;
    buf[4] = 14;

    buf.resize(3);

    CUDA_ASSERT_EQ(buf.size(), static_cast<std::size_t>(3));
    CUDA_EXPECT_EQ(buf[0], 10);
    CUDA_EXPECT_EQ(buf[1], 11);
    CUDA_EXPECT_EQ(buf[2], 12);
}

CUDA_TEST(HostBuffer, MoveConstructorTransfersOwnership) {

    HostBuffer<int> a;
    a.resize(3);

    a[0] = 21;
    a[1] = 22;
    a[2] = 23;

    const HostBuffer<int> b(std::move(a));

    CUDA_ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    CUDA_EXPECT_EQ(b[0], 21);
    CUDA_EXPECT_EQ(b[1], 22);
    CUDA_EXPECT_EQ(b[2], 23);

    CUDA_EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

CUDA_TEST(HostBuffer, MoveAssignmentTransfersOwnership) {

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

    CUDA_ASSERT_EQ(b.size(), static_cast<std::size_t>(3));
    CUDA_EXPECT_EQ(b[0], 31);
    CUDA_EXPECT_EQ(b[1], 32);
    CUDA_EXPECT_EQ(b[2], 33);

    CUDA_EXPECT_EQ(a.size(), static_cast<std::size_t>(0));
}

CUDA_TEST(HostBuffer, ResizeToZeroClearsSize) {

    HostBuffer<int> buf;
    buf.resize(4);

    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    buf[3] = 4;

    buf.resize(0);

    CUDA_EXPECT_EQ(buf.size(), static_cast<std::size_t>(0));
}
