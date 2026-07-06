#include <atlas/memory/copy.h>

#include <array>
#include <cstddef>
#include <gtest/gtest.h>

TEST(Copy, CopyHostToDeviceRawPointersCopiesValues) {
    std::array<int, 8> src {};
    std::array<int, 8> dst {};

    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int>(10 + i);
        dst[i] = -1;
    }

    atlas::copy_host_to_device(src.data(), dst.data(), src.size());

    for (std::size_t i = 0; i < dst.size(); ++i) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(Copy, CopyDeviceToHostRawPointersCopiesValues) {
    std::array<float, 6> src {};
    std::array<float, 6> dst {};

    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = 0.25f * static_cast<float>(i + 1);
        dst[i] = -99.0f;
    }

    atlas::copy_device_to_host(src.data(), dst.data(), src.size());

    for (std::size_t i = 0; i < dst.size(); ++i) {
        EXPECT_FLOAT_EQ(dst[i], src[i]);
    }
}

TEST(Copy, DeviceBufferRoundTripPreservesValues) {
    constexpr std::array<int, 4> src { 1, 2, 3, 4 };
    atlas::DeviceBuffer<int> buffer(src.size());

    atlas::copy_host_to_device(src.data(), buffer, src.size());

    std::array<int, 4> dst { -7, -7, -7, -7 };
    atlas::copy_device_to_host(buffer, dst.data(), dst.size());

    for (std::size_t i = 0; i < dst.size(); ++i) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(Copy, CopyHostToDeviceCountZeroDoesNotTouchDestination) {
    constexpr std::array<int, 4> src { 1, 2, 3, 4 };
    std::array<int, 4> dst { -7, -7, -7, -7 };

    atlas::copy_host_to_device(src.data(), dst.data(), 0);

    for (const int value : dst) {
        EXPECT_EQ(value, -7);
    }
}

TEST(Copy, CopyDeviceToHostCountZeroDoesNotTouchDestination) {
    constexpr std::array<float, 5> src { 1.f, 2.f, 3.f, 4.f, 5.f };
    std::array<float, 5> dst { -3.f, -3.f, -3.f, -3.f, -3.f };

    atlas::copy_device_to_host(src.data(), dst.data(), 0);

    for (const float value : dst) {
        EXPECT_FLOAT_EQ(value, -3.f);
    }
}
