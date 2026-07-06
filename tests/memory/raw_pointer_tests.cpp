#include <atlas/memory/raw_pointer_cast.h>

#include <atlas/buffer/device_buffer.h>
#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <type_traits>

TEST(RawPointerCast, DevicePtrIsThrustDevicePtrAlias) {
    EXPECT_TRUE((std::is_same_v<atlas::device_ptr<int>, thrust::device_ptr<int>>));
    EXPECT_TRUE((std::is_same_v<atlas::device_ptr<const int>, thrust::device_ptr<const int>>));
}

TEST(RawPointerCast, RawPointerCastIdentityNonConst) {
    int x  = 123;
    int* p = &x;

    int* rp = atlas::raw_pointer_cast(p);

    EXPECT_EQ(rp, p);
    EXPECT_EQ(*rp, 123);

    *rp = 77;
    EXPECT_EQ(x, 77);
}

TEST(RawPointerCast, RawPointerCastIdentityConst) {
    constexpr int x = 42;
    const int* p    = &x;

    const int* rp = atlas::raw_pointer_cast(p);

    EXPECT_EQ(rp, p);
    EXPECT_EQ(*rp, 42);
}

TEST(RawPointerCast, RawPointerCastUnwrapsDevicePtr) {
    atlas::DeviceBuffer<int> values(1, -5);

    const atlas::device_ptr<int> dp = values.data();
    int* rp                         = atlas::raw_pointer_cast(dp);

    EXPECT_EQ(rp, thrust::raw_pointer_cast(values.data()));
}

TEST(RawPointerCast, RawPointerCastUnwrapsConstDevicePtr) {
    const atlas::DeviceBuffer<int> values(1, 9);

    const atlas::device_ptr<const int> dp = values.data();
    const int* rp                         = atlas::raw_pointer_cast(dp);

    EXPECT_EQ(rp, thrust::raw_pointer_cast(values.data()));
}
