#include "../../utilities/test_utils.h"

#include <atlas/collider/kernel/dt_remain_collider_kernel.h>
#include <atlas/geometry/plane.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

namespace {

using atlas::IsothermalSurfaceInteraction;
using atlas::Plane;
using atlas::Sync;
using atlas::Unit;
using atlas::Vector3F;
using atlas::DtRemainColliderKernel;
using atlas::test::vec_near;
using atlas::tol;

Unit<float>
make_plane_unit() {
    static const auto geometry = Plane<float>::builder()
                                     .with_point_normal(Vector3F(0, 0, 0), Vector3F(1, 0, 0))
                                     .make_host_shared();
    const auto sync = Sync<float>::builder()
                          .make_host_shared();

    return Unit<float>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

IsothermalSurfaceInteraction<float>
make_specular_interaction() {
    return IsothermalSurfaceInteraction<float>::builder()
        .with_restitution(1.0f)
        .with_momentum_acc(0.0f)
        .build();
}

} // namespace

TEST(DtRemainColliderKernel, SweepMotionUsesIncidentVelocityOverTimeStep) {
    const DtRemainColliderKernel<float> kernel;
    Vector3F sweep_direction {};
    float sweep_speed {};
    float sweep_length {};

    kernel.sweep_motion(
        make_plane_unit(),
        Vector3F(0.0f, 0.0f, 0.0f),
        Vector3F(2.0f, 0.0f, 0.0f),
        2.0f,
        0.5f,
        sweep_direction,
        sweep_speed,
        sweep_length);

    EXPECT_TRUE(vec_near(sweep_direction, Vector3F(1.0f, 0.0f, 0.0f), tol));
    EXPECT_NEAR(sweep_speed, 2.0f, tol);
    EXPECT_NEAR(sweep_length, 1.0f, tol);
}

TEST(DtRemainColliderKernel, OperatorMovesForRemainingTimeAfterVelocityUpdate) {
    const DtRemainColliderKernel<float> kernel;
    Vector3F position(-1.0f, 0.0f, 0.0f);
    Vector3F velocity(2.0f, 0.0f, 0.0f);

    kernel(
        position,
        velocity,
        Vector3F(2.0f, 0.0f, 0.0f),
        Vector3F(0.0f, 0.0f, 0.0f),
        Vector3F(1.0f, 0.0f, 0.0f),
        1.0f,
        2.0f,
        1.0f,
        make_plane_unit(),
        make_specular_interaction());

    EXPECT_TRUE(vec_near(position, Vector3F(-1.0f + static_cast<float>(tol), 0.0f, 0.0f), tol));
    EXPECT_TRUE(vec_near(velocity, Vector3F(-2.0f, 0.0f, 0.0f), tol));
}

TEST(DtRemainColliderKernel, OperatorStopsAtOffsetWhenSweepSpeedIsDegenerate) {
    const DtRemainColliderKernel<float> kernel;
    Vector3F position(-1.0f, 0.0f, 0.0f);
    Vector3F velocity(2.0f, 0.0f, 0.0f);

    kernel(
        position,
        velocity,
        Vector3F(2.0f, 0.0f, 0.0f),
        Vector3F(0.0f, 0.0f, 0.0f),
        Vector3F(1.0f, 0.0f, 0.0f),
        1.0f,
        0.0f,
        1.0f,
        make_plane_unit(),
        make_specular_interaction());

    EXPECT_TRUE(vec_near(position, Vector3F(static_cast<float>(tol), 0.0f, 0.0f), tol));
    EXPECT_TRUE(vec_near(velocity, Vector3F(-2.0f, 0.0f, 0.0f), tol));
}
