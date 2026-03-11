#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>

#include <array>
#include <cstddef>
#include <gtest/gtest.h>

#if defined(ATLAS_TASKING_TBB)

TEST(Copy_TBB, CopyHostToDevice_RawPointersCopiesBytes) {
    std::array<int, 8> src {};
    std::array<int, 8> dst {};
    std::array<int, 8> exp {};

    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int>(10 + i);
        dst[i] = -1;
        exp[i] = src[i];
    }

    atlas::copy_host_to_device(src.data(), dst.data(), src.size());

    for (std::size_t i = 0; i < dst.size(); ++i) {
        EXPECT_EQ(dst[i], exp[i]);
    }
}

TEST(Copy_TBB, CopyDeviceToHost_RawPointersCopiesBytes) {
    std::array<double, 6> src {};
    std::array<double, 6> dst {};
    std::array<double, 6> exp {};

    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = 0.25 * static_cast<double>(i + 1);
        dst[i] = -99.0;
        exp[i] = src[i];
    }

    atlas::copy_device_to_host(src.data(), dst.data(), src.size());

    for (std::size_t i = 0; i < dst.size(); ++i) {
        EXPECT_DOUBLE_EQ(dst[i], exp[i]);
    }
}

TEST(Copy_TBB, CopyHostToDevice_CountZeroDoesNotTouchDestination) {
    constexpr std::array<int, 4> src { 1, 2, 3, 4 };
    std::array<int, 4> dst { -7, -7, -7, -7 };

    atlas::copy_host_to_device(src.data(), dst.data(), 0);

    for (int i : dst) {
        EXPECT_EQ(i, -7);
    }
}

TEST(Copy_TBB, CopyDeviceToHost_CountZeroDoesNotTouchDestination) {
    constexpr std::array<float, 5> src { 1.f, 2.f, 3.f, 4.f, 5.f };
    std::array<float, 5> dst { -3.f, -3.f, -3.f, -3.f, -3.f };

    atlas::copy_device_to_host(src.data(), dst.data(), 0);

    for (const float i : dst) {
        EXPECT_FLOAT_EQ(i, -3.f);
    }
}

#else

TEST(Copy_TBB, SkippedBecauseNotTBBBackend) {
    GTEST_SKIP() << "ATLAS_TASKING_TBB is not enabled.";
}

#endif