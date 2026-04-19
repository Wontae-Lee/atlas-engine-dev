#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/sync/sync_operator.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;
using Quat = atlas::Quaternion<T>;
using Ray = atlas::spatial::Ray<T>;

constexpr T kEps = static_cast<T>(1e-4);

} // namespace

TEST(SyncOperator, DefaultConstructorCreatesIdentityTransform) {
    const atlas::SyncOperator<T> sync;

    EXPECT_TRUE(atlas::test::vec_near(sync.translation, Vec3(0, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(sync.sync_to_world(Vec3(1, 2, 3)), Vec3(1, 2, 3), kEps));
}

TEST(SyncOperator, PointTransformsRoundTrip) {
    const atlas::SyncOperator<T> sync(Vec3(1, 2, 3), Quat::from_axis_angle(Vec3(0, 0, 1), T(atlas::pi / 2)));

    const Vec3 local(1, 0, 0);
    const Vec3 world = sync.sync_to_world(local);
    const Vec3 roundtrip = sync.sync_to_local(world);

    EXPECT_TRUE(atlas::test::vec_near(roundtrip, local, 1e-3f));
}

TEST(SyncOperator, DirectionTransformsIgnoreTranslation) {
    const atlas::SyncOperator<T> sync(Vec3(5, 6, 7), Quat::from_axis_angle(Vec3(0, 0, 1), T(atlas::pi / 2)));

    const Vec3 world_dir = sync.sync_dir_to_world(Vec3(1, 0, 0));

    EXPECT_NEAR(world_dir.length(), 1.0f, 1e-3f);
    EXPECT_TRUE(atlas::test::vec_near(world_dir, Vec3(0, 1, 0), 1e-3f));
}

TEST(SyncOperator, RayTransformsRoundTrip) {
    const atlas::SyncOperator<T> sync(Vec3(1, 0, 0), Quat::from_axis_angle(Vec3(0, 1, 0), T(atlas::pi / 2)));
    const Ray local(Vec3(0, 0, 0), Vec3(0, 0, 1));

    const Ray world = sync.sync_to_world(local);
    const Ray restored = sync.sync_to_local(world);

    EXPECT_TRUE(atlas::test::vec_near(restored.origin, local.origin, 1e-3f));
    EXPECT_TRUE(atlas::test::vec_near(restored.direction, local.direction, 1e-3f));
}
