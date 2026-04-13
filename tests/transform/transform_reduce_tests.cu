#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

#include <vector>

CUDA_TEST(TransformReduce, EmptyRangeReturnsInit_Serial) {
    const std::vector<int> v;

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        7,
        ::atlas::test::cuda::SquareOp<int> {},
        ::atlas::test::cuda::PlusOp<int> {});

    CUDA_EXPECT_EQ(r, 7);
}

CUDA_TEST(TransformReduce, EmptyRangeReturnsInit_Host) {
    const std::vector<int> v;

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::host>(
        v.begin(),
        v.end(),
        -3,
        ::atlas::test::cuda::IncrementOp<int> {},
        ::atlas::test::cuda::PlusOp<int> {});

    CUDA_EXPECT_EQ(r, -3);
}

CUDA_TEST(TransformReduce, EmptyRangeReturnsInit_Device) {
    const std::vector<int> v;

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::device>(
        v.begin(),
        v.end(),
        123,
        ::atlas::test::cuda::IdentityOp<int> {},
        ::atlas::test::cuda::PlusOp<int> {});

    CUDA_EXPECT_EQ(r, 123);
}

CUDA_TEST(TransformReduce, SumSquares_MatchesExpected_Serial) {
    const std::vector<int> v { 1, 2, 3, 4 };

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        0,
        ::atlas::test::cuda::SquareOp<int> {},
        ::atlas::test::cuda::PlusOp<int> {});

    CUDA_EXPECT_EQ(r, 30);
}

CUDA_TEST(TransformReduce, SumSquares_HostMatchesSerial) {
    const std::vector<int> v { 1, 2, 3, 4, 5, 6, 7 };

    const int serial_r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        0,
        ::atlas::test::cuda::SquareOp<int> {},
        ::atlas::test::cuda::PlusOp<int> {});

    const int host_r = atlas::transform_reduce<atlas::ExecutionPolicy::host>(
        v.begin(),
        v.end(),
        0,
        ::atlas::test::cuda::SquareOp<int> {},
        ::atlas::test::cuda::PlusOp<int> {});

    CUDA_EXPECT_EQ(host_r, serial_r);
}

CUDA_TEST(TransformReduce, SumSquares_DeviceMatchesSerial) {
    const std::vector<int> v { 3, 1, 4, 1, 5, 9 };

    const int serial_r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        0,
        ::atlas::test::cuda::SquareOp<int> {},
        ::atlas::test::cuda::PlusOp<int> {});

    const int device_r = atlas::transform_reduce<atlas::ExecutionPolicy::device>(
        v.begin(),
        v.end(),
        0,
        ::atlas::test::cuda::SquareOp<int> {},
        ::atlas::test::cuda::PlusOp<int> {});

    CUDA_EXPECT_EQ(device_r, serial_r);
}

CUDA_TEST(TransformReduce, NonZeroInit_IsIncluded) {
    const std::vector<int> v { 1, 2, 3 };

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        10,
        ::atlas::test::cuda::IdentityOp<int> {},
        ::atlas::test::cuda::PlusOp<int> {});

    CUDA_EXPECT_EQ(r, 16);
}

CUDA_TEST(TransformReduce, MultiplyReduce_WorksForAssociativeOp) {
    const std::vector<int> v { 1, 2, 3, 4 };

    const int serial_r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        1,
        ::atlas::test::cuda::IncrementOp<int> {},
        ::atlas::test::cuda::MultiplyOp<int> {});

    CUDA_EXPECT_EQ(serial_r, 120);

    const int host_r = atlas::transform_reduce<atlas::ExecutionPolicy::host>(
        v.begin(),
        v.end(),
        1,
        ::atlas::test::cuda::IncrementOp<int> {},
        ::atlas::test::cuda::MultiplyOp<int> {});

    CUDA_EXPECT_EQ(host_r, serial_r);

    const int device_r = atlas::transform_reduce<atlas::ExecutionPolicy::device>(
        v.begin(),
        v.end(),
        1,
        ::atlas::test::cuda::IncrementOp<int> {},
        ::atlas::test::cuda::MultiplyOp<int> {});

    CUDA_EXPECT_EQ(device_r, serial_r);
}
