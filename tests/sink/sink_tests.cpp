#include "../utilities/test_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/geometry/box.h>
#include <atlas/sink/sink.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::HostBuffer;
using atlas::Sink;
using atlas::Unit;
using atlas::Vector3F;
using atlas::fluid::DespawnOperator;
using atlas::fluid::DespawnType;
using atlas::physics::Sync;

FluidHostPtr<float>
make_fluid() {
    return Fluid<float>::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

Unit<float>
make_unit() {
    const auto geometry = Box<float>::builder()
                              .with_lower_corner(Vector3F(-1, -1, -1))
                              .with_upper_corner(Vector3F(1, 1, 1))
                              .make_host_shared();

    const auto sync = Sync<float>::builder()
                          .make_host_shared();

    return Unit<float>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

DespawnOperator<float>
make_despawn_operator() {
    return DespawnOperator<float>(DespawnType::Surface);
}

} // namespace

TEST(Sink, BuilderConstructsUsableSink) {
    // Arrange: create a fluid and fully configured sink.
    const auto fluid = make_fluid();

    auto sink = Sink<float>::builder()
                    .with_units(HostBuffer<Unit<float>> { make_unit() })
                    .with_fluid(fluid)
                    .with_despawn_types(HostBuffer<DespawnType> { DespawnType::Surface })
                    .with_despawn_operator(make_despawn_operator())
                    .with_tolerance(0.1f)
                    .with_flip(true)
                    .build();

    // Assert: update and direct sink execution are callable on a valid sink.
    EXPECT_NO_THROW(sink.update(0.1f));
    EXPECT_NO_THROW(sink.sink());
}

TEST(Sink, BuilderRejectsMissingDependencies) {
    // Arrange: create the shared fluid dependency.
    const auto fluid = make_fluid();

    // A sink cannot be built without units.
    EXPECT_THROW(Sink<float>::builder()
                     .with_fluid(fluid)
                     .with_despawn_types(HostBuffer<DespawnType> { DespawnType::Surface })
                     .with_despawn_operator(make_despawn_operator())
                     .build(),
                 std::runtime_error);

    // A sink cannot be built without a fluid.
    EXPECT_THROW(Sink<float>::builder()
                     .with_units(HostBuffer<Unit<float>> { make_unit() })
                     .with_despawn_types(HostBuffer<DespawnType> { DespawnType::Surface })
                     .with_despawn_operator(make_despawn_operator())
                     .build(),
                 std::runtime_error);
}

TEST(Sink, BuilderRejectsMismatchedDespawnConfigurationSizes) {
    // Arrange: create the shared fluid dependency.
    const auto fluid = make_fluid();

    // Despawn operator count must match unit count.
    EXPECT_THROW(Sink<float>::builder()
                     .with_units(HostBuffer<Unit<float>> { make_unit(), make_unit() })
                     .with_fluid(fluid)
                     .with_despawn_types(HostBuffer<DespawnType> { DespawnType::Surface })
                     .with_despawn_operators(HostBuffer<DespawnOperator<float>> { make_despawn_operator(), make_despawn_operator(), make_despawn_operator() })
                     .build(),
                 std::runtime_error);

    // Despawn type count must match unit count.
    EXPECT_THROW(Sink<float>::builder()
                     .with_units(HostBuffer<Unit<float>> { make_unit(), make_unit() })
                     .with_fluid(fluid)
                     .with_despawn_types(HostBuffer<DespawnType> { DespawnType::Surface, DespawnType::Volume, DespawnType::Surface })
                     .with_despawn_operator(make_despawn_operator())
                     .build(),
                 std::runtime_error);
}

TEST(Sink, BuilderRejectsInvalidImmediateInputs) {
    // Empty unit lists are rejected immediately.
    EXPECT_THROW(Sink<float>::builder()
                     .with_units(HostBuffer<Unit<float>> {}),
                 std::runtime_error);

    // Empty despawn type lists are rejected immediately.
    EXPECT_THROW(Sink<float>::builder()
                     .with_despawn_types(HostBuffer<DespawnType> {}),
                 std::runtime_error);

    // Empty despawn operator lists are rejected immediately.
    EXPECT_THROW(Sink<float>::builder()
                     .with_despawn_operators(HostBuffer<DespawnOperator<float>> {}),
                 std::runtime_error);
}

TEST(Sink, MakeHostSharedBuildsSink) {
    // Arrange: create the shared fluid dependency.
    const auto fluid = make_fluid();

    // Act: build a sink through host shared ownership.
    const auto sink = Sink<float>::builder()
                          .with_units(HostBuffer<Unit<float>> { make_unit() })
                          .with_fluid(fluid)
                          .with_despawn_types(HostBuffer<DespawnType> { DespawnType::Surface })
                          .with_despawn_operator(make_despawn_operator())
                          .make_host_shared();

    // Assert: the shared sink exists and accepts updates.
    ASSERT_NE(sink, nullptr);
    EXPECT_NO_THROW(sink->update(0.1f));
}

TEST(Sink, UpdateIgnoresNonPositiveDt) {
    // Arrange: create a valid sink.
    const auto fluid = make_fluid();

    auto sink = Sink<float>::builder()
                    .with_units(HostBuffer<Unit<float>> { make_unit() })
                    .with_fluid(fluid)
                    .with_despawn_types(HostBuffer<DespawnType> { DespawnType::Surface })
                    .with_despawn_operator(make_despawn_operator())
                    .build();

    // Assert: non-positive dt is ignored without throwing.
    EXPECT_NO_THROW(sink.update(0.0f));
}
