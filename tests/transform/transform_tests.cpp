#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <vector>

TEST(Transform, Unary_EmptyRangeReturnsDestUnchanged) {
    std::vector<int> in;
    std::vector<int> out;

    auto* out0    = out.data();
    const auto it = atlas::transform<atlas::ExecutionPolicy::serial>(in.begin(), in.end(), out.begin(), [](const int x) { return x + 1; });

    EXPECT_EQ(it, out.begin());
    EXPECT_EQ(out.data(), out0);
}

TEST(Transform, Unary_SerialAppliesOpAndReturnsEnd) {
    std::vector<int> in { 1, 2, 3, 4 };
    std::vector<int> out(in.size(), 0);

    const auto it = atlas::transform<atlas::ExecutionPolicy::serial>(
        in.begin(),
        in.end(),
        out.begin(),
        [](const int x) { return x * 2; });

    EXPECT_EQ(it, out.begin() + static_cast<std::ptrdiff_t>(in.size()));

    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 4);
    EXPECT_EQ(out[2], 6);
    EXPECT_EQ(out[3], 8);
}

TEST(Transform, Unary_HostMatchesSerial) {
    std::vector<int> in { 1, 2, 3, 4, 5 };
    std::vector<int> a(in.size(), 0);
    std::vector<int> b(in.size(), 0);

    (void)atlas::transform<atlas::ExecutionPolicy::serial>(
        in.begin(),
        in.end(),
        a.begin(),
        [](const int x) { return x + 7; });

    (void)atlas::transform<atlas::ExecutionPolicy::host>(
        in.begin(),
        in.end(),
        b.begin(),
        [](const int x) { return x + 7; });

    EXPECT_EQ(a, b);
}

TEST(Transform, Unary_DeviceMatchesSerial) {
    std::vector<int> in { 3, 1, 4, 1, 5, 9 };
    std::vector<int> a(in.size(), 0);
    std::vector<int> b(in.size(), 0);

    (void)atlas::transform<atlas::ExecutionPolicy::serial>(
        in.begin(),
        in.end(),
        a.begin(),
        [](const int x) { return x - 2; });

    (void)atlas::transform<atlas::ExecutionPolicy::device>(
        in.begin(),
        in.end(),
        b.begin(),
        [](const int x) { return x - 2; });

    EXPECT_EQ(a, b);
}

TEST(Transform, Binary_SerialAppliesOpAndReturnsEnd) {
    std::vector<int> x { 1, 2, 3, 4 };
    std::vector<int> y { 10, 20, 30, 40 };
    std::vector<int> out(x.size(), 0);

    const auto it = atlas::transform<atlas::ExecutionPolicy::serial>(
        x.begin(),
        x.end(),
        y.begin(),
        out.begin(),
        [](const int a, const int b) { return a + b; });

    EXPECT_EQ(it, out.begin() + static_cast<std::ptrdiff_t>(x.size()));

    EXPECT_EQ(out[0], 11);
    EXPECT_EQ(out[1], 22);
    EXPECT_EQ(out[2], 33);
    EXPECT_EQ(out[3], 44);
}

TEST(Transform, Binary_HostMatchesSerial) {
    std::vector<int> x { 2, 4, 6, 8, 10 };
    std::vector<int> y { 1, 3, 5, 7, 9 };
    std::vector<int> a(x.size(), 0);
    std::vector<int> b(x.size(), 0);

    (void)atlas::transform<atlas::ExecutionPolicy::serial>(
        x.begin(),
        x.end(),
        y.begin(),
        a.begin(),
        [](const int u, const int v) { return u - v; });

    (void)atlas::transform<atlas::ExecutionPolicy::host>(
        x.begin(),
        x.end(),
        y.begin(),
        b.begin(),
        [](const int u, const int v) { return u - v; });

    EXPECT_EQ(a, b);
}

TEST(Transform, Binary_DeviceMatchesSerial) {
    std::vector<int> x { 1, 1, 2, 3, 5, 8 };
    std::vector<int> y { 0, 1, 1, 2, 3, 5 };
    std::vector<int> a(x.size(), 0);
    std::vector<int> b(x.size(), 0);

    (void)atlas::transform<atlas::ExecutionPolicy::serial>(
        x.begin(),
        x.end(),
        y.begin(),
        a.begin(),
        [](const int u, const int v) { return u * 10 + v; });

    (void)atlas::transform<atlas::ExecutionPolicy::device>(
        x.begin(),
        x.end(),
        y.begin(),
        b.begin(),
        [](const int u, const int v) { return u * 10 + v; });

    EXPECT_EQ(a, b);
}

TEST(Transform, Unary_AllowsInPlaceTransformForSerial) {

    std::vector<int> v { 1, 2, 3, 4 };

    (void)atlas::transform<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        v.begin(),
        [](const int x) { return x * x; });

    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 4);
    EXPECT_EQ(v[2], 9);
    EXPECT_EQ(v[3], 16);
}