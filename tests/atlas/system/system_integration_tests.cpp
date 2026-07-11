#include <atlas/system/system.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/material/material.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/math.h>
#include <atlas/sink/sink.h>
#include <atlas/sink/volume_sink.h>
#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/solver.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

// End-to-end tests that drive the full System::update() pipeline. On the TBB
// backend every stage's parallel_for runs on the host, so these exercise the
// real advect, mark_survivors, and compact kernels through the driver rather
// than any single unit under test.
namespace {

using atlas::Box;
using atlas::DeviceBuffer;
using atlas::DsmcKernelType;
using atlas::DsmcSolver;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::FluidPositionState;
using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::Float3;
using atlas::Material;
using atlas::MaterialDictionary;
using atlas::Molecule;
using atlas::UniverseCollisionCountState;
using atlas::Geometry;
using atlas::HostBuffer;
using atlas::Quaternion;
using atlas::Sink;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::System;
using atlas::Unit;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::VolumeSink;

UniverseHostPtr
make_universe_ptr() {
    return Universe::builder().make_host_unique();
}

// A fluid whose position and velocity columns are seeded from the host. The
// buffer is sized to hold at least the given particles so emission head-room
// never reallocates mid-test.
FluidHostPtr
make_fluid(const std::vector<Float3>& positions,
           const std::vector<Float3>& velocities,
           const std::size_t buffer_size) {
    const std::size_t n = positions.size();

    FluidHostPtr fluid = Fluid::builder()
                             .with_buffer_size(buffer_size == 0 ? 1 : buffer_size)
                             .with_particle_count(n)
                             .make_host_unique();

    const HostBuffer<Float3> host_positions(positions.begin(), positions.end());
    fluid->state<FluidPositionState>()->data()
        = DeviceBuffer<Float3>(host_positions.begin(), host_positions.end());

    const HostBuffer<Float3> host_velocities(velocities.begin(), velocities.end());
    fluid->state<FluidVelocityState>()->data()
        = DeviceBuffer<Float3>(host_velocities.begin(), host_velocities.end());

    return fluid;
}

// An axis-aligned static volume sink that despawns any particle inside [lower, upper].
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

std::vector<Float3>
read_positions(const System& system, const std::size_t count) {
    const auto& data = system.fluid()->state<FluidPositionState>()->data();
    const HostBuffer<Float3> host(data.begin(), data.end());

    return std::vector<Float3>(host.begin(), host.begin() + static_cast<std::ptrdiff_t>(count));
}

std::vector<Float3>
read_velocities(const System& system, const std::size_t count) {
    const auto& data = system.fluid()->state<FluidVelocityState>()->data();
    const HostBuffer<Float3> host(data.begin(), data.end());

    return std::vector<Float3>(host.begin(), host.begin() + static_cast<std::ptrdiff_t>(count));
}

// Sum of the per-cell candidate-collision counters the DSMC solver writes this step.
long long
total_collision_count(const System& system) {
    const auto* state = system.universe()->state<UniverseCollisionCountState>();
    if (state == nullptr) {
        return -1;
    }

    const auto& data = state->data();
    const HostBuffer<int> host(data.begin(), data.end());

    long long total = 0;
    for (const int count : host) {
        total += count;
    }
    return total;
}

// A single-cell universe packed with fast N2 molecules and wired to a VHS DSMC
// solver — no source or sink, so particle count is conserved and any change to
// the fluid comes from collisions or advection alone.
System
make_dense_dsmc_system(const std::size_t particle_count) {
    std::vector<Float3> positions;
    std::vector<Float3> velocities;
    positions.reserve(particle_count);
    velocities.reserve(particle_count);

    for (std::size_t i = 0; i < particle_count; ++i) {
        const float f = static_cast<float>(i);
        positions.push_back(Float3(0.2f + 0.01f * f, 0.3f + 0.005f * f, 0.5f));
        velocities.push_back(Float3(i % 2 ? -300.0f : 300.0f,
                                    i % 3 ? 200.0f : -200.0f,
                                    i % 2 ? 100.0f : -100.0f));
    }

    // Nitrogen (N2) as a VHS species: mass, three energy stubs, reference diameter,
    // reference temperature, viscosity index, scattering parameter.
    auto materials = MaterialDictionary::builder()
                         .with_material(Material(Molecule(4.65e-26f, 0.0f, 0.0f, 0.0f,
                                                          4.17e-10f, 273.0f, 0.74f, 1.0f)))
                         .make_host_shared();

    FluidHostPtr fluid = Fluid::builder()
                             .with_buffer_size(particle_count)
                             .with_particle_count(particle_count)
                             .with_statistical_weight(1.0e18f)
                             .with_materials(materials)
                             .make_host_unique();

    const HostBuffer<Float3> host_positions(positions.begin(), positions.end());
    fluid->state<FluidPositionState>()->data()
        = DeviceBuffer<Float3>(host_positions.begin(), host_positions.end());

    const HostBuffer<Float3> host_velocities(velocities.begin(), velocities.end());
    fluid->state<FluidVelocityState>()->data()
        = DeviceBuffer<Float3>(host_velocities.begin(), host_velocities.end());

    const HostBuffer<std::size_t> host_species(particle_count, 0);
    fluid->state<FluidSpeciesState>()->data()
        = DeviceBuffer<std::size_t>(host_species.begin(), host_species.end());

    UniverseHostPtr universe = Universe::builder()
                                   .with_lower_corner(Float3(0.0f, 0.0f, 0.0f))
                                   .with_upper_corner(Float3(1.0f, 1.0f, 1.0f))
                                   .with_cell_size(1.0f)
                                   .make_host_unique();

    auto solver = DsmcSolver::builder()
                      .with_kernel_type(DsmcKernelType::variable_hard_sphere)
                      .make_host_shared();

    return System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_solver(solver)
        .with_dt(1.0e-4f)
        .build();
}

}

TEST(SystemIntegration, AdvectionIntegratesPositionByVelocityTimesDt) {
    const float dt = 0.5f;

    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) },
                                               { Float3(2.0f, -1.0f, 0.0f) },
                                               1))
                        .with_universe(make_universe_ptr())
                        .with_dt(dt)
                        .build();

    system.update();

    const std::vector<Float3> after = read_positions(system, 1);
    EXPECT_NEAR(after[0].x, 1.0f, 1.0e-5f);   // 0 + 2.0 * 0.5
    EXPECT_NEAR(after[0].y, -0.5f, 1.0e-5f);  // 0 + -1.0 * 0.5
    EXPECT_NEAR(after[0].z, 0.0f, 1.0e-5f);
}

TEST(SystemIntegration, AZeroVelocityParticleStaysPutAcrossAStep) {
    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(3.0f, 4.0f, 5.0f) },
                                               { Float3(0.0f, 0.0f, 0.0f) },
                                               1))
                        .with_universe(make_universe_ptr())
                        .with_dt(1.0f)
                        .build();

    system.update();

    const std::vector<Float3> after = read_positions(system, 1);
    EXPECT_FLOAT_EQ(after[0].x, 3.0f);
    EXPECT_FLOAT_EQ(after[0].y, 4.0f);
    EXPECT_FLOAT_EQ(after[0].z, 5.0f);
}

TEST(SystemIntegration, RepeatedStepsAccumulateDisplacement) {
    const float dt = 0.25f;

    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) },
                                               { Float3(1.0f, 0.0f, 0.0f) },
                                               1))
                        .with_universe(make_universe_ptr())
                        .with_dt(dt)
                        .build();

    for (int step = 0; step < 4; ++step) {
        system.update();
    }

    // Four steps of 1.0 * 0.25 each.
    const std::vector<Float3> after = read_positions(system, 1);
    EXPECT_NEAR(after[0].x, 1.0f, 1.0e-5f);
}

TEST(SystemIntegration, SinkRemovesTheParticleInsideItsVolumeDuringUpdate) {
    // Two stationary particles: one inside the sink box, one well outside it.
    System system = System::builder()
                        .with_fluid(make_fluid(
                            { Float3(0.0f, 0.0f, 0.0f), Float3(10.0f, 0.0f, 0.0f) },
                            { Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f) },
                            2))
                        .with_universe(make_universe_ptr())
                        .with_sink(make_volume_sink(Float3(-1.0f, -1.0f, -1.0f),
                                                    Float3(1.0f, 1.0f, 1.0f)))
                        .with_dt(1.0f)
                        .build();

    ASSERT_EQ(system.fluid()->particle_count(), 2u);

    system.update();

    // Only the particle at the origin was inside the sink volume; compaction keeps
    // the survivor at index 0.
    ASSERT_EQ(system.fluid()->particle_count(), 1u);
    const std::vector<Float3> survivors = read_positions(system, 1);
    EXPECT_FLOAT_EQ(survivors[0].x, 10.0f);
}

TEST(SystemIntegration, AnAdvectingParticleIsRemovedOnceItEntersTheSink) {
    // A particle drifts along +x at 1 unit/step toward a sink box centered near x = 5.
    System system = System::builder()
                        .with_fluid(make_fluid({ Float3(0.0f, 0.0f, 0.0f) },
                                               { Float3(1.0f, 0.0f, 0.0f) },
                                               1))
                        .with_universe(make_universe_ptr())
                        .with_sink(make_volume_sink(Float3(4.5f, -1.0f, -1.0f),
                                                    Float3(5.5f, 1.0f, 1.0f)))
                        .with_dt(1.0f)
                        .build();

    ASSERT_EQ(system.fluid()->particle_count(), 1u);

    // It is still outside the sink for the first few steps, then crosses into it.
    for (int step = 0; step < 4; ++step) {
        system.update();
        EXPECT_EQ(system.fluid()->particle_count(), 1u) << "removed too early at step " << step;
    }

    // By step ~5 the particle has reached x = 5, inside the sink, and is despawned.
    for (int step = 0; step < 4 && system.fluid()->particle_count() > 0; ++step) {
        system.update();
    }
    EXPECT_EQ(system.fluid()->particle_count(), 0u);
}

TEST(SystemIntegration, DsmcPipelineConservesParticleCountAndKeepsStateFinite) {
    // search -> allocate -> solve -> advect all run on the TBB backend here. With no
    // source or sink, the closed system must neither gain nor lose particles, and no
    // collision or integration step may produce a non-finite position or velocity.
    const std::size_t particle_count = 60;
    System system = make_dense_dsmc_system(particle_count);

    ASSERT_EQ(system.fluid()->particle_count(), particle_count);

    for (int step = 0; step < 15; ++step) {
        system.update();
        ASSERT_EQ(system.fluid()->particle_count(), particle_count)
            << "particle count changed at step " << step;
    }

    const std::vector<Float3> positions  = read_positions(system, particle_count);
    const std::vector<Float3> velocities = read_velocities(system, particle_count);
    for (std::size_t i = 0; i < particle_count; ++i) {
        ASSERT_TRUE(std::isfinite(positions[i].x) && std::isfinite(positions[i].y)
                    && std::isfinite(positions[i].z))
            << "non-finite position at particle " << i;
        ASSERT_TRUE(std::isfinite(velocities[i].x) && std::isfinite(velocities[i].y)
                    && std::isfinite(velocities[i].z))
            << "non-finite velocity at particle " << i;
    }
}

TEST(SystemIntegration, DsmcSolverRecordsCollisionsInADenseCell) {
    // The NTC scheme is deterministic (its draws are hashed from fixed seeds), so a
    // dense, fast cell reproducibly produces candidate collisions across the run.
    System system = make_dense_dsmc_system(60);

    long long collisions_over_run = 0;
    for (int step = 0; step < 15; ++step) {
        system.update();
        const long long step_collisions = total_collision_count(system);
        ASSERT_GE(step_collisions, 0) << "collision-count state missing at step " << step;
        collisions_over_run += step_collisions;
    }

    EXPECT_GT(collisions_over_run, 0);
}
