#include "../utilities/test_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/sync/sync.h>

#include <testkit/testkit.h>

namespace {

using atlas::pi;
using atlas::QuaternionF;
using atlas::Sync;
using atlas::tol;
using atlas::Vector3F;
using atlas::test::vec_near;

} // namespace

TEST(Sync, DefaultConstructorCreatesIdentityTransform) {
    // Arrange: create a default sync object.
    const Sync<float> sync;

    // Assert: the default transform leaves points unchanged.
    EXPECT_TRUE(vec_near(sync.sync_to_world(Vector3F(1, 2, 3)), Vector3F(1, 2, 3), tol));
}

TEST(Sync, BuilderConstructsConfiguredSync) {
    // Arrange and act: build a sync object with translation and a +Z rotation.
    const auto sync = Sync<float>::builder()
                          .with_rigid_pose(Vector3F(1, 2, 3),
                                           QuaternionF::from_axis_angle(Vector3F(0, 0, 1), static_cast<float>(pi / 2)))
                          .build();

    // Assert: the configured rigid transform is applied to local points.
    EXPECT_TRUE(vec_near(sync.sync_to_world(Vector3F(1, 0, 0)), Vector3F(1, 3, 3), tol));
}

TEST(Sync, BuilderRejectsInvalidQuaternion) {
    // A zero-length quaternion cannot define a valid orientation.
    EXPECT_THROW(Sync<float>::builder()
                     .with_rigid_pose(Vector3F(0, 0, 0), QuaternionF(0, 0, 0, 0))
                     .build(),
                 std::runtime_error);
}

TEST(Sync, SettersUpdatePose) {
    // Arrange: start from the identity transform.
    Sync<float> sync;

    // Act: update translation and orientation through setters.
    sync.set_translation(Vector3F(1, 2, 3));
    sync.set_orientation(QuaternionF::from_axis_angle(Vector3F(0, 0, 1), static_cast<float>(pi / 2)));

    // Assert: the updated pose is reflected in point transforms.
    EXPECT_TRUE(vec_near(sync.sync_to_world(Vector3F(1, 0, 0)), Vector3F(1, 3, 3), tol));
}

TEST(Sync, MakeHostSharedBuildsSync) {
    // Arrange and act: build shared ownership for an identity sync object.
    const auto sync = Sync<float>::builder()
                          .with_rigid_pose(Vector3F(0, 0, 0), QuaternionF(1, 0, 0, 0))
                          .make_host_shared();

    // Assert: the shared object exists and preserves identity transforms.
    ASSERT_NE(sync, nullptr);
    EXPECT_TRUE(vec_near(sync->sync_to_world(Vector3F(1, 0, 0)), Vector3F(1, 0, 0), tol));
}
