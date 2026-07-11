#include <atlas/system/system.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/generator/generator.h>
#include <atlas/sink/sink.h>
#include <atlas/sink/volume_sink.h>
#include <atlas/solver/solver.h>
#include <atlas/source/source.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using atlas::Box;
using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::FluidPositionState;
using atlas::FluidVelocityState;
using atlas::Float3;
using atlas::GeneratorHostPtr;
using atlas::Geometry;
using atlas::HostBuffer;
using atlas::Quaternion;
using atlas::Sink;
using atlas::SolverHostPtr;
using atlas::SourceHostPtr;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::System;
using atlas::Unit;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::VolumeSink;

/** @brief A fresh unit-box universe handle for wiring into a system. */
UniverseHostPtr
make_universe_ptr() {
    return Universe::builder().make_host_unique();
}

/**
 * @brief Builds a fluid of @p positions particles with those positions and zero velocity.
 *
 * The mandatory position and velocity columns are the ones @c mark_survivors reads, so
 * both are seeded from the host before the fluid is handed to a system.
 */
FluidHostPtr
make_fluid(const std::vector<Float3>& positions) {
    const std::size_t n = positions.size();

    FluidHostPtr fluid = Fluid::builder()
                             .with_buffer_size(n == 0 ? 1 : n)
                             .with_particle_count(n)
                             .make_host_unique();

    const HostBuffer<Float3> host_positions(positions.begin(), positions.end());
    fluid->state<FluidPositionState>()->data()
        = DeviceBuffer<Float3>(host_positions.begin(), host_positions.end());

    const HostBuffer<Float3> host_velocities(n, Float3(0.0f, 0.0f, 0.0f));
    fluid->state<FluidVelocityState>()->data()
        = DeviceBuffer<Float3>(host_velocities.begin(), host_velocities.end());

    return fluid;
}

/** @brief An axis-aligned static volume sink: it despawns any particle inside [lower, upper]. */
Sink
make_volume_sink(const Float3& lower, const Float3& upper) {
    const SyncHostPtr sync = Sync::builder()
                                 .with_rigid_pose(Float3(0.0f, 0.0f, 0.0f),
                                                  Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
                                 .make_host_shared();

    Unit unit = Unit::builder()
                    .with_geometry(Geometry(Box::builder()
                                                .with_lower_corner(lower)
                                                .with_upper_corner(upper)
                                                .build()))
                    .with_sync(sync)
                    .build();

    return Sink(VolumeSink::builder().with_unit(std::move(unit)).build());
}

/** @brief Copies the leading @p count survivor flags to the host. */
std::vector<int>
read_active(const System& system, const std::size_t count) {
    const auto& active = system.fluid()->active();
    const HostBuffer<int> host(active.begin(), active.end());

    return std::vector<int>(host.begin(), host.begin() + static_cast<std::ptrdiff_t>(count));
}

/** @brief Copies the leading @p count particle positions to the host. */
std::vector<Float3>
read_positions(const System& system, const std::size_t count) {
    const auto& data = system.fluid()->state<FluidPositionState>()->data();
    const HostBuffer<Float3> host(data.begin(), data.end());

    return std::vector<Float3>(host.begin(), host.begin() + static_cast<std::ptrdiff_t>(count));
}

}

TEST(System, BuilderRejectsMissingFluid) {
    EXPECT_THROW(
        static_cast<void>(System::builder().with_universe(make_universe_ptr()).with_dt(0.01f).build()),
        std::runtime_error);
}

TEST(System, BuilderRejectsMissingUniverse) {
    EXPECT_THROW(
        static_cast<void>(System::builder().with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) })).with_dt(0.01f).build()),
        std::runtime_error);
}

TEST(System, BuilderRejectsNonPositiveTimestep) {
    EXPECT_THROW(
        static_cast<void>(System::builder()
                              .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) }))
                              .with_universe(make_universe_ptr())
                              .with_dt(0.0f)
                              .build()),
        std::runtime_error);
}

TEST(System, FreshSystemReportsZeroStepAndConfiguredCounts) {
    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) }))
                        .with_universe(make_universe_ptr())
                        .with_sink(make_volume_sink(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f)))
                        .with_dt(0.02f)
                        .build();

    EXPECT_EQ(system.step(), 0u);
    EXPECT_EQ(system.source_count(), 0u);
    EXPECT_EQ(system.solver_count(), 0u);
    EXPECT_EQ(system.collider_count(), 0u);
    EXPECT_EQ(system.sink_count(), 1u);
    EXPECT_FLOAT_EQ(system.dt(), 0.02f);
}

TEST(System, MarkSurvivorsFlagsOnlyParticlesInsideTheSink) {
    // Five particles on the x-axis; the sink box covers x in [1.5, 3.5].
    const std::vector<Float3> positions {
        Float3(0.0f, 0.0f, 0.0f),
        Float3(1.0f, 0.0f, 0.0f),
        Float3(2.0f, 0.0f, 0.0f),
        Float3(3.0f, 0.0f, 0.0f),
        Float3(4.0f, 0.0f, 0.0f),
    };

    System system = System::builder()
                        .with_fluid(make_fluid(positions))
                        .with_universe(make_universe_ptr())
                        .with_sink(make_volume_sink(Float3(1.5f, -1.0f, -1.0f), Float3(3.5f, 1.0f, 1.0f)))
                        .with_dt(0.01f)
                        .build();

    system.mark_survivors(static_cast<int>(positions.size()));

    // x = 2 and x = 3 fall inside the sink and are marked dead (0); the rest survive.
    const std::vector<int> active = read_active(system, positions.size());
    EXPECT_EQ(active, (std::vector<int> { 1, 1, 0, 0, 1 }));
}

TEST(System, CompactionKeepsExactlyTheSurvivorsInOrder) {
    const std::vector<Float3> positions {
        Float3(0.0f, 0.0f, 0.0f),
        Float3(1.0f, 0.0f, 0.0f),
        Float3(2.0f, 0.0f, 0.0f),
        Float3(3.0f, 0.0f, 0.0f),
        Float3(4.0f, 0.0f, 0.0f),
    };

    System system = System::builder()
                        .with_fluid(make_fluid(positions))
                        .with_universe(make_universe_ptr())
                        .with_sink(make_volume_sink(Float3(1.5f, -1.0f, -1.0f), Float3(3.5f, 1.0f, 1.0f)))
                        .with_dt(0.01f)
                        .build();

    system.mark_survivors(static_cast<int>(positions.size()));
    const std::size_t kept = (*system.fluid()).compact();

    ASSERT_EQ(kept, 3u);
    EXPECT_EQ(system.fluid()->particle_count(), 3u);

    // Survivors x = 0, 1, 4 remain, in their original relative order.
    const std::vector<Float3> survivors = read_positions(system, kept);
    ASSERT_EQ(survivors.size(), 3u);
    EXPECT_FLOAT_EQ(survivors[0].x, 0.0f);
    EXPECT_FLOAT_EQ(survivors[1].x, 1.0f);
    EXPECT_FLOAT_EQ(survivors[2].x, 4.0f);
}

TEST(System, AllParticlesSurviveWhenTheSinkClaimsNone) {
    const std::vector<Float3> positions {
        Float3(0.0f, 0.0f, 0.0f),
        Float3(1.0f, 0.0f, 0.0f),
        Float3(2.0f, 0.0f, 0.0f),
    };

    System system = System::builder()
                        .with_fluid(make_fluid(positions))
                        .with_universe(make_universe_ptr())
                        // Sink far from every particle.
                        .with_sink(make_volume_sink(Float3(100.0f, 100.0f, 100.0f), Float3(101.0f, 101.0f, 101.0f)))
                        .with_dt(0.01f)
                        .build();

    system.mark_survivors(static_cast<int>(positions.size()));
    EXPECT_EQ(read_active(system, positions.size()), (std::vector<int> { 1, 1, 1 }));

    const std::size_t kept = (*system.fluid()).compact();
    EXPECT_EQ(kept, positions.size());

    // Positions are untouched when nothing is removed.
    const std::vector<Float3> remaining = read_positions(system, kept);
    ASSERT_EQ(remaining.size(), 3u);
    EXPECT_FLOAT_EQ(remaining[2].x, 2.0f);
}

TEST(System, AllParticlesDieWhenTheSinkClaimsEveryone) {
    const std::vector<Float3> positions {
        Float3(0.0f, 0.0f, 0.0f),
        Float3(1.0f, 0.0f, 0.0f),
        Float3(2.0f, 0.0f, 0.0f),
    };

    System system = System::builder()
                        .with_fluid(make_fluid(positions))
                        .with_universe(make_universe_ptr())
                        // Sink covering the whole extent.
                        .with_sink(make_volume_sink(Float3(-10.0f, -10.0f, -10.0f), Float3(10.0f, 10.0f, 10.0f)))
                        .with_dt(0.01f)
                        .build();

    system.mark_survivors(static_cast<int>(positions.size()));
    EXPECT_EQ(read_active(system, positions.size()), (std::vector<int> { 0, 0, 0 }));

    const std::size_t kept = (*system.fluid()).compact();
    EXPECT_EQ(kept, 0u);
    EXPECT_EQ(system.fluid()->particle_count(), 0u);
}

TEST(System, MarkSurvivorsAndCompactAreNoOpsOnAnEmptyParticleSet) {
    System system = System::builder()
                        .with_fluid(make_fluid({}))
                        .with_universe(make_universe_ptr())
                        .with_sink(make_volume_sink(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f)))
                        .with_dt(0.01f)
                        .build();

    ASSERT_EQ(system.fluid()->particle_count(), 0u);

    // No particles to test; neither call must crash or produce survivors.
    system.mark_survivors(0);
    const std::size_t kept = (*system.fluid()).compact();

    EXPECT_EQ(kept, 0u);
    EXPECT_EQ(system.fluid()->particle_count(), 0u);
}

TEST(System, RecordSpawnedIsANoOpWithoutAnObserver) {
    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) }))
                        .with_universe(make_universe_ptr())
                        .with_dt(0.01f)
                        .build();

    ASSERT_FALSE(static_cast<bool>(system.observer()));

    // Without an observer the counter path is skipped entirely; this must not crash.
    system.record_spawned(0, 0, 1);

    SUCCEED();
}

TEST(SystemBuilder, RejectsNullSolver) {
    EXPECT_THROW(
        static_cast<void>(System::builder()
                              .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) }))
                              .with_universe(make_universe_ptr())
                              .with_solver(SolverHostPtr {})
                              .with_dt(0.01f)
                              .build()),
        std::runtime_error);
}

TEST(SystemBuilder, RejectsNullEmitterEntries) {
    // with_emitter keeps the source/generator lists aligned, but validate() rejects a
    // null entry in either list at build time.
    EXPECT_THROW(
        static_cast<void>(System::builder()
                              .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) }))
                              .with_universe(make_universe_ptr())
                              .with_emitter(SourceHostPtr {}, GeneratorHostPtr {})
                              .with_dt(0.01f)
                              .build()),
        std::runtime_error);
}

TEST(SystemBuilder, BuildSucceedsWithTheDefaultTimestep) {
    // With no with_dt() call the builder's positive default (0.01) passes validation.
    const System system = System::builder()
                              .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) }))
                              .with_universe(make_universe_ptr())
                              .build();

    EXPECT_FLOAT_EQ(system.dt(), 0.01f);
    EXPECT_EQ(system.step(), 0u);
}

TEST(System, EmitIsANoOpWithoutSources) {
    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f) }))
                        .with_universe(make_universe_ptr())
                        .with_dt(0.01f)
                        .build();

    ASSERT_EQ(system.source_count(), 0u);

    // No sources means no spawns; the live particle count is unchanged.
    system.emit();

    EXPECT_EQ(system.fluid()->particle_count(), 2u);
}

TEST(System, AllocateIsANoOpWithoutCodec) {
    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) }))
                        .with_universe(make_universe_ptr())
                        .with_dt(0.01f)
                        .build();

    ASSERT_FALSE(static_cast<bool>(system.codec()));

    // Without a codec the allocation phase returns immediately.
    system.allocate();

    SUCCEED();
}

TEST(System, SolveIsANoOpWithoutSolvers) {
    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) }))
                        .with_universe(make_universe_ptr())
                        .with_dt(0.01f)
                        .build();

    ASSERT_EQ(system.solver_count(), 0u);

    // With no solvers the solve phase returns without touching the fluid.
    system.solve();

    EXPECT_EQ(system.fluid()->particle_count(), 1u);
}

TEST(System, RemoveIsANoOpWithoutSinks) {
    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f) }))
                        .with_universe(make_universe_ptr())
                        .with_dt(0.01f)
                        .build();

    ASSERT_EQ(system.sink_count(), 0u);

    // No sinks means nothing is despawned and no compaction occurs.
    system.remove();

    EXPECT_EQ(system.fluid()->particle_count(), 2u);
}

TEST(System, SnapshotDirectoryNameFormatsTheStep) {
    EXPECT_EQ(System::snapshot_directory_name(0), "time_step_0");
    EXPECT_EQ(System::snapshot_directory_name(42), "time_step_42");
}
