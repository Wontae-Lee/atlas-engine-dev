#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>

#include <gtest/gtest.h>

#if !defined(ATLAS_TASKING_CUDA)

TEST(RawPointerCast_CPU, DevicePtrIsRawPointerAlias) {
    EXPECT_TRUE((std::is_same_v<atlas::device_ptr<int>, int*>));
    EXPECT_TRUE((std::is_same_v<atlas::device_ptr<const int>, const int*>));
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
    int x                           = -5;
    const atlas::device_ptr<int> dp = &x;

    int* rp = atlas::raw_pointer_cast(dp);

    EXPECT_EQ(rp, &x);
    EXPECT_EQ(*rp, -5);
}

#else

TEST(RawPointerCast_CPU, SkippedBecauseCudaBackend) {
    GTEST_SKIP() << "CPU-only raw_pointer_cast overloads are not active when ATLAS_TASKING_CUDA is enabled.";
}

#endif