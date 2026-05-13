#include "../utilities/test_utils.h"

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <testkit/testkit.h>

TEST(RawPointerCast_CPU, DevicePtrIsRawPointerAlias) {
#if defined(ATLAS_TASKING_CUDA)
    EXPECT_FALSE((std::is_same_v<atlas::device_ptr<int>, int*>));
    EXPECT_FALSE((std::is_same_v<atlas::device_ptr<const int>, const int*>));
#else
    EXPECT_TRUE((std::is_same_v<atlas::device_ptr<int>, int*>));
    EXPECT_TRUE((std::is_same_v<atlas::device_ptr<const int>, const int*>));
#endif
}

TEST(RawPointerCast_CPU, RawPointerCastIdentity_NonConst) {
    int x  = 123;
    int* p = &x;

    int* rp = atlas::raw_pointer_cast(p);

    EXPECT_EQ(rp, p);
    EXPECT_EQ(*rp, 123);

    *rp = 77;
    EXPECT_EQ(x, 77);
}

TEST(RawPointerCast_CPU, RawPointerCastIdentity_Const) {
    constexpr int x = 42;
    const int* p    = &x;

    const int* rp = atlas::raw_pointer_cast(p);

    EXPECT_EQ(rp, p);
    EXPECT_EQ(*rp, 42);
}

TEST(RawPointerCast_CPU, RawPointerCastWorksWithDevicePtrAlias) {
#if defined(ATLAS_TASKING_CUDA)
    thrust::device_vector<int> values(1);
    const int expected = -5;
    atlas::copy_host_to_device(&expected, values, 1);

    const atlas::device_ptr<int> dp = values.data();
    int* rp = atlas::raw_pointer_cast(dp);
    int actual = 0;
    atlas::copy_device_to_host(rp, &actual, 1);

    EXPECT_EQ(rp, thrust::raw_pointer_cast(values.data()));
    EXPECT_EQ(actual, expected);
#else
    int x                           = -5;
    const atlas::device_ptr<int> dp = &x;

    int* rp = atlas::raw_pointer_cast(dp);

    EXPECT_EQ(rp, &x);
    EXPECT_EQ(*rp, -5);
#endif
}
