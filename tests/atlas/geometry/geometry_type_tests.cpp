#include <atlas/geometry/geometry_type.h>

#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::GeometryType;
using atlas::GeometryVariant;

/// A discriminant value that is not one of the eight registered enumerators, used
/// to exercise the normalize/fallback path.
constexpr GeometryType bogus_tag = static_cast<GeometryType>(999);

} // namespace

/**
 * The tag is copied to the device inside a trivially-copyable Geometry, so it must
 * have a stable, fixed int underlying type.
 */
TEST(GeometryType, HasFixedIntUnderlyingType) {
    static_assert(std::is_same_v<std::underlying_type_t<GeometryType>, int>,
                  "GeometryType must have a fixed int underlying type for device buffers");

    EXPECT_TRUE((std::is_same_v<std::underlying_type_t<GeometryType>, int>));
}

/**
 * The enumerators carry the default 0..7 values in declaration order.
 */
TEST(GeometryType, EnumeratorsAreDeclarationOrdered) {
    EXPECT_EQ(static_cast<int>(GeometryType::box), 0);
    EXPECT_EQ(static_cast<int>(GeometryType::circle), 1);
    EXPECT_EQ(static_cast<int>(GeometryType::cylinder), 2);
    EXPECT_EQ(static_cast<int>(GeometryType::plane), 3);
    EXPECT_EQ(static_cast<int>(GeometryType::sphere), 4);
    EXPECT_EQ(static_cast<int>(GeometryType::square), 5);
    EXPECT_EQ(static_cast<int>(GeometryType::triangle), 6);
    EXPECT_EQ(static_cast<int>(GeometryType::triangle_mesh), 7);
}

/**
 * Casting a tag to its underlying int and back reproduces the same enumerator.
 */
TEST(GeometryType, RoundTripsThroughUnderlyingInt) {
    const GeometryType tags[] = {
        GeometryType::box,      GeometryType::circle,   GeometryType::cylinder,
        GeometryType::plane,    GeometryType::sphere,   GeometryType::square,
        GeometryType::triangle, GeometryType::triangle_mesh,
    };

    for (const GeometryType tag : tags) {
        const int value = static_cast<int>(tag);
        EXPECT_EQ(static_cast<GeometryType>(value), tag);
    }
}

/**
 * Every registered tag is recognized by the variant's case list.
 */
TEST(GeometryType, VariantContainsEveryRegisteredTag) {
    EXPECT_TRUE(GeometryVariant::contains(GeometryType::box));
    EXPECT_TRUE(GeometryVariant::contains(GeometryType::circle));
    EXPECT_TRUE(GeometryVariant::contains(GeometryType::cylinder));
    EXPECT_TRUE(GeometryVariant::contains(GeometryType::plane));
    EXPECT_TRUE(GeometryVariant::contains(GeometryType::sphere));
    EXPECT_TRUE(GeometryVariant::contains(GeometryType::square));
    EXPECT_TRUE(GeometryVariant::contains(GeometryType::triangle));
    EXPECT_TRUE(GeometryVariant::contains(GeometryType::triangle_mesh));

    EXPECT_FALSE(GeometryVariant::contains(bogus_tag));
}

/**
 * An unknown tag normalizes to the documented fallback, `sphere`, while a valid tag
 * normalizes to itself.
 */
TEST(GeometryType, UnknownTagNormalizesToSphere) {
    EXPECT_EQ(GeometryVariant::normalize(bogus_tag), GeometryType::sphere);
    EXPECT_EQ(GeometryVariant::normalize(GeometryType::box), GeometryType::box);
    EXPECT_EQ(GeometryVariant::normalize(GeometryType::sphere), GeometryType::sphere);
}
