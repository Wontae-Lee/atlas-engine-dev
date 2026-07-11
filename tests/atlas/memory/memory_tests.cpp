#include <atlas/memory/memory.h>

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace {

using atlas::make_device_shared;
using atlas::make_host_shared;
using atlas::make_host_unique;

/// A small payload whose two-argument constructor exercises perfect forwarding
/// through the factory helpers.
struct Payload {
    Payload(const int a, const double b) : a(a), b(b) { }
    int a;
    double b;
};

// Every alias in this header collapses to a std::shared_ptr / std::unique_ptr on a
// CPU-only build, and make_device_shared allocates managed storage under CUDA; the
// cases below drive only the surface that is identical on both backends
// (construction with forwarded arguments, dereference, and ownership counts).

}

TEST(MakeHostShared, ConstructsWithForwardedArgumentsAndSingleOwner) {
    const std::shared_ptr<Payload> p = make_host_shared<Payload>(7, 2.5);

    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->a, 7);
    EXPECT_DOUBLE_EQ(p->b, 2.5);
    EXPECT_EQ(p.use_count(), 1L);
}

TEST(MakeHostShared, DecaysTheRequestedTypeToAPlainSharedPtr) {
    // The factory strips references/const so the owned type is the decayed form.
    const auto p = make_host_shared<int>(42);
    static_assert(std::is_same_v<decltype(p), const std::shared_ptr<int>>);
    EXPECT_EQ(*p, 42);
}

TEST(MakeHostShared, SharesOwnershipOnCopy) {
    const std::shared_ptr<Payload> a = make_host_shared<Payload>(1, 1.0);
    const std::shared_ptr<Payload> b = a;

    EXPECT_EQ(a.use_count(), 2L);
    EXPECT_EQ(a.get(), b.get());
}

TEST(MakeHostUnique, ConstructsWithForwardedArguments) {
    const std::unique_ptr<Payload> p = make_host_unique<Payload>(-3, 8.25);

    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->a, -3);
    EXPECT_DOUBLE_EQ(p->b, 8.25);
}

TEST(MakeHostUnique, DecaysTheRequestedTypeToAPlainUniquePtr) {
    const auto p = make_host_unique<double>(1.5);
    static_assert(std::is_same_v<decltype(p), const std::unique_ptr<double>>);
    EXPECT_DOUBLE_EQ(*p, 1.5);
}

TEST(MakeDeviceShared, ConstructsAnOwnedObjectFromForwardedArguments) {
    // On a CPU build this is a std::shared_ptr; under CUDA a managed device_shared_ptr.
    // Both expose get(), operator*, use_count(), and a bool test.
    const auto p = make_device_shared<Payload>(11, 4.0);

    ASSERT_NE(p.get(), nullptr);
    EXPECT_TRUE(static_cast<bool>(p));
    EXPECT_EQ((*p).a, 11);
    EXPECT_DOUBLE_EQ(p->b, 4.0);
    EXPECT_EQ(p.use_count(), 1);
}
