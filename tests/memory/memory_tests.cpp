#include <atlas/memory/memory.h>

#include <gtest/gtest.h>

namespace {

struct Payload {
    int value;

    explicit Payload(const int v)
        : value(v) { }
};

}

TEST(Memory, MakeHostSharedConstructsPayload) {
    const atlas::host_shared_ptr<Payload> ptr = atlas::make_host_shared<Payload>(41);

    ASSERT_TRUE(ptr != nullptr);
    EXPECT_EQ(ptr->value, 41);
    EXPECT_EQ(ptr.use_count(), 1);
}

TEST(Memory, MakeDeviceSharedConstructsPayload) {
    const atlas::device_shared_ptr<Payload> ptr = atlas::make_device_shared<Payload>(7);

    ASSERT_TRUE(static_cast<bool>(ptr));
    EXPECT_EQ(ptr->value, 7);
    EXPECT_EQ(ptr.use_count(), 1);
}

TEST(Memory, DeviceSharedCopyIncrementsUseCount) {
    const atlas::device_shared_ptr<Payload> first = atlas::make_device_shared<Payload>(3);

    {
        const atlas::device_shared_ptr<Payload> second = first;
        EXPECT_EQ(first.use_count(), 2);
        EXPECT_EQ(second->value, 3);
    }

    EXPECT_EQ(first.use_count(), 1);
}
