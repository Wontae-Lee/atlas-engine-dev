#include <atlas/math/vector/bool3.h>

#include <gtest/gtest.h>

namespace {

using atlas::all;
using atlas::any;
using atlas::Bool3;
using atlas::none;

}

TEST(Bool3, AllIsTrueOnlyWhenEveryAxisIsTrue) {
    EXPECT_TRUE(all(Bool3 { true, true, true }));
    EXPECT_FALSE(all(Bool3 { true, false, true }));
    EXPECT_FALSE(all(Bool3 { false, false, false }));
}

TEST(Bool3, AnyIsTrueWhenAtLeastOneAxisIsTrue) {
    EXPECT_TRUE(any(Bool3 { false, true, false }));
    EXPECT_TRUE(any(Bool3 { true, true, true }));
    EXPECT_FALSE(any(Bool3 { false, false, false }));
}

TEST(Bool3, NoneIsTheNegationOfAny) {
    EXPECT_TRUE(none(Bool3 { false, false, false }));
    EXPECT_FALSE(none(Bool3 { false, false, true }));
}

TEST(Bool3, AndIsComponentWise) {
    const Bool3 r = Bool3 { true, true, false } & Bool3 { true, false, false };
    EXPECT_TRUE(r.x);
    EXPECT_FALSE(r.y);
    EXPECT_FALSE(r.z);
}

TEST(Bool3, OrIsComponentWise) {
    const Bool3 r = Bool3 { true, false, false } | Bool3 { false, false, true };
    EXPECT_TRUE(r.x);
    EXPECT_FALSE(r.y);
    EXPECT_TRUE(r.z);
}

TEST(Bool3, NotFlipsEveryAxis) {
    const Bool3 r = !Bool3 { true, false, true };
    EXPECT_FALSE(r.x);
    EXPECT_TRUE(r.y);
    EXPECT_FALSE(r.z);
}
