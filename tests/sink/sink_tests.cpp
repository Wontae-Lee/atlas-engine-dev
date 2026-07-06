#include <atlas/sink/sink.h>

#include <atlas/fluid/fluid.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/box.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace {

using atlas::Box;
using atlas::Despawn;
using atlas::DespawnType;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::HostBuffer;
using atlas::Sink;
using atlas::Sync;
using atlas::Unit;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

FluidHostPtr
make_fluid() {
    return Fluid::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

UniverseHostPtr
make_universe(const HostBuffer<Unit>& sink_units) {
    return Universe::builder()
        .with_lower_corner(Vector3(-10.0f, -10.0f, -10.0f))
        .with_upper_corner(Vector3(10.0f, 10.0f, 10.0f))
        .with_cell_size(1.0f)
        .with_sink_units(sink_units)
        .make_host_shared();
}

Unit
make_unit() {
    static const auto geometry = Box::builder()
                                     .with_lower_corner(Vector3(-1.0f, -1.0f, -1.0f))
                                     .with_upper_corner(Vector3(1.0f, 1.0f, 1.0f))
                                     .make_host_shared();

    const auto sync = Sync::builder()
                          .make_host_shared();

    return Unit::builder()
        .with_geometry(atlas::Geometry(*geometry))
        .with_sync(sync)
        .build();
}

Unit
make_tracing_unit() {
    static const auto geometry = Box::builder()
                                     .with_lower_corner(Vector3(2.0f, -1.0f, -1.0f))
                                     .with_upper_corner(Vector3(3.0f, 1.0f, 1.0f))
                                     .make_host_shared();

    const auto sync = Sync::builder()
                          .make_host_shared();

    return Unit::builder()
        .with_geometry(atlas::Geometry(*geometry))
        .with_sync(sync)
        .build();
}

Despawn
make_despawn_operator() {
    return Despawn(DespawnType::surface);
}

}

TEST(Sink, BuilderConstructsUsableSink) {
    const auto fluid = make_fluid();

    auto sink = Sink::builder()
                    .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                    .with_fluid(fluid)
                    .with_despawn_types(HostBuffer<DespawnType> { DespawnType::surface })
                    .with_despawn_operator(make_despawn_operator())
                    .with_tolerance(0.1f)
                    .with_flip(true)
                    .build();

    EXPECT_NO_THROW(sink.update(0.1f));
    EXPECT_NO_THROW(sink.sink());
}

TEST(Sink, BuilderRejectsMissingDependencies) {
    const auto fluid = make_fluid();

    EXPECT_THROW(static_cast<void>(Sink::builder()
                                       .with_fluid(fluid)
                                       .with_despawn_types(HostBuffer<DespawnType> { DespawnType::surface })
                                       .with_despawn_operator(make_despawn_operator())
                                       .build()),
                 std::runtime_error);

    EXPECT_THROW(static_cast<void>(Sink::builder()
                                       .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                                       .with_despawn_types(HostBuffer<DespawnType> { DespawnType::surface })
                                       .with_despawn_operator(make_despawn_operator())
                                       .build()),
                 std::runtime_error);

    EXPECT_THROW(static_cast<void>(Sink::builder()
                                       .with_universe(make_universe(HostBuffer<Unit> {}))
                                       .with_fluid(fluid)
                                       .with_despawn_types(HostBuffer<DespawnType> { DespawnType::surface })
                                       .with_despawn_operator(make_despawn_operator())
                                       .build()),
                 std::runtime_error);
}

TEST(Sink, BuilderRejectsMismatchedDespawnConfigurationSizes) {
    const auto fluid = make_fluid();

    EXPECT_THROW(static_cast<void>(Sink::builder()
                                       .with_universe(make_universe(HostBuffer<Unit> { make_unit(), make_unit() }))
                                       .with_fluid(fluid)
                                       .with_despawn_types(HostBuffer<DespawnType> { DespawnType::surface })
                                       .with_despawn_operators(HostBuffer<Despawn> { make_despawn_operator(), make_despawn_operator(), make_despawn_operator() })
                                       .build()),
                 std::runtime_error);

    EXPECT_THROW(static_cast<void>(Sink::builder()
                                       .with_universe(make_universe(HostBuffer<Unit> { make_unit(), make_unit() }))
                                       .with_fluid(fluid)
                                       .with_despawn_types(HostBuffer<DespawnType> { DespawnType::surface, DespawnType::volume, DespawnType::surface })
                                       .with_despawn_operator(make_despawn_operator())
                                       .build()),
                 std::runtime_error);
}

TEST(Sink, BuilderRejectsInvalidImmediateInputs) {
    EXPECT_THROW(Sink::builder()
                     .with_despawn_types(HostBuffer<DespawnType> {}),
                 std::runtime_error);

    EXPECT_THROW(Sink::builder()
                     .with_despawn_operators(HostBuffer<Despawn> {}),
                 std::runtime_error);
}

TEST(Sink, MakeHostSharedBuildsSink) {
    const auto fluid = make_fluid();

    const auto sink = Sink::builder()
                          .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                          .with_fluid(fluid)
                          .with_despawn_types(HostBuffer<DespawnType> { DespawnType::surface })
                          .with_despawn_operator(make_despawn_operator())
                          .make_host_shared();

    ASSERT_NE(sink, nullptr);
    EXPECT_NO_THROW(sink->update(0.1f));
}

TEST(Sink, UpdateIgnoresNonPositiveDt) {
    const auto fluid = make_fluid();

    auto sink = Sink::builder()
                    .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                    .with_fluid(fluid)
                    .with_despawn_types(HostBuffer<DespawnType> { DespawnType::surface })
                    .with_despawn_operator(make_despawn_operator())
                    .build();

    EXPECT_NO_THROW(sink.update(0.0f));
}

TEST(Sink, TracingDespawnUsesPositionVelocityAndUpdateDt) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(2);

    auto* position_state = fluid->state<atlas::FluidPositionState>();
    auto* velocity_state = fluid->state<atlas::FluidVelocityState>();
    auto* active_state   = fluid->state<atlas::FluidActiveState>();
    ASSERT_NE(position_state, nullptr);
    ASSERT_NE(velocity_state, nullptr);
    ASSERT_NE(active_state, nullptr);

    position_state->data()[0] = Vector3(1.0f, 0.0f, 0.0f);
    position_state->data()[1] = Vector3(-2.0f, 0.0f, 0.0f);
    velocity_state->data()[0] = Vector3(1.0f, 0.0f, 0.0f);
    velocity_state->data()[1] = Vector3(1.0f, 0.0f, 0.0f);
    active_state->data()[0]   = 1;
    active_state->data()[1]   = 1;

    auto sink = Sink::builder()
                    .with_universe(make_universe(HostBuffer<Unit> { make_tracing_unit() }))
                    .with_fluid(fluid)
                    .with_despawn_types(HostBuffer<DespawnType> { DespawnType::tracing })
                    .with_despawn_operator(Despawn(DespawnType::tracing))
                    .build();

    sink.update(2.5f);

    EXPECT_EQ(fluid->particle_count(), std::size_t { 1 });
    const Vector3 remaining_position = position_state->data()[0];
    expect_vec_near(remaining_position, Vector3(-2.0f, 0.0f, 0.0f));
}
