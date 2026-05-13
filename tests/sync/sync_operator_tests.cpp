#include "../utilities/test_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/sync/sync_operator.h>

#include <testkit/testkit.h>

namespace {

using atlas::QuaternionF;
using atlas::RayF;
using atlas::SyncOperator;
using atlas::Vector3F;
using atlas::pi;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(SyncOperator, DefaultConstructorCreatesIdentityTransform) {
    // Arrange: create a default sync operator.
    const SyncOperator<float> sync;

    // Assert: default translation and point transform are identity.
    EXPECT_TRUE(vec_near(sync.translation, Vector3F(0, 0, 0), tol));
    EXPECT_TRUE(vec_near(sync.sync_to_world(Vector3F(1, 2, 3)), Vector3F(1, 2, 3), tol));
}

TEST(SyncOperator, PointTransformsRoundTrip) {
    // Arrange: create a rigid transform with translation and a +Z rotation.
    const SyncOperator<float> sync(
        Vector3F(1, 2, 3),
        QuaternionF::from_axis_angle(Vector3F(0, 0, 1), static_cast<float>(pi / 2)));

    // Act: transform a point to world space and back.
    const Vector3F local(1, 0, 0);
    const Vector3F world     = sync.sync_to_world(local);
    const Vector3F roundtrip = sync.sync_to_local(world);

    // Assert: point transforms round-trip through the inverse transform.
    EXPECT_TRUE(vec_near(roundtrip, local, tol));
}

TEST(SyncOperator, DirectionTransformsIgnoreTranslation) {
    // Arrange: create a transform with non-zero translation and rotation.
    const SyncOperator<float> sync(
        Vector3F(5, 6, 7),
        QuaternionF::from_axis_angle(Vector3F(0, 0, 1), static_cast<float>(pi / 2)));

    // Act: transform a local direction to world space.
    const Vector3F world_dir = sync.sync_dir_to_world(Vector3F(1, 0, 0));

    // Assert: only orientation affects directions.
    EXPECT_NEAR(world_dir.length(), 1.0f, tol);
    EXPECT_TRUE(vec_near(world_dir, Vector3F(0, 1, 0), tol));
}

TEST(SyncOperator, RayTransformsRoundTrip) {
    // Arrange: create a rigid transform and a local ray.
    const SyncOperator<float> sync(
        Vector3F(1, 0, 0),
        QuaternionF::from_axis_angle(Vector3F(0, 1, 0), static_cast<float>(pi / 2)));
    const RayF local(Vector3F(0, 0, 0), Vector3F(0, 0, 1));

    // Act: transform the ray to world space and back.
    const RayF world    = sync.sync_to_world(local);
    const RayF restored = sync.sync_to_local(world);

    // Assert: both ray origin and direction round-trip.
    EXPECT_TRUE(vec_near(restored.origin, local.origin, tol));
    EXPECT_TRUE(vec_near(restored.direction, local.direction, tol));
}
