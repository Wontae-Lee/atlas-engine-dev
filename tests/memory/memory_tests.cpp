#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>

#include <gtest/gtest.h>

#if !defined(ATLAS_TASKING_CUDA)

TEST(SharedPtr_CPU, AliasesAreStdSharedPtr) {
    EXPECT_TRUE((std::is_same_v<atlas::host_shared_ptr<int>, std::shared_ptr<int>>));
    EXPECT_TRUE((std::is_same_v<atlas::device_shared_ptr<int>, std::shared_ptr<int>>));
}

TEST(SharedPtr_CPU, MakeHostSharedConstructsObject) {
    const auto p = atlas::make_host_shared<atlas::test::Foo>(7, 2.5);

    ASSERT_TRUE(static_cast<bool>(p));
    EXPECT_EQ(p->x, 7);
    EXPECT_DOUBLE_EQ(p->y, 2.5);
}

TEST(SharedPtr_CPU, MakeDeviceSharedConstructsObject) {
    const auto p = atlas::make_device_shared<atlas::test::Foo>(3, -1.25);

    ASSERT_TRUE(static_cast<bool>(p));
    EXPECT_EQ(p->x, 3);
    EXPECT_DOUBLE_EQ(p->y, -1.25);
}

TEST(SharedPtr_CPU, SharedOwnershipWorks) {
    const auto p0 = atlas::make_host_shared<atlas::test::Foo>(1, 1.0);
    ASSERT_TRUE(static_cast<bool>(p0));
    EXPECT_EQ(p0.use_count(), 1);

    auto p1 = p0;
    EXPECT_EQ(p0.use_count(), 2);
    EXPECT_EQ(p1.use_count(), 2);

    p1.reset();
    EXPECT_EQ(p0.use_count(), 1);
}

#else

TEST(SharedPtr_CPU, SkippedBecauseCudaBackend) {
    GTEST_SKIP() << "CPU-only shared_ptr fallback is not active when ATLAS_TASKING_CUDA is enabled.";
}

#endif