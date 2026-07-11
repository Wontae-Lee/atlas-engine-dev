/**
 * @file main.cpp
 * @brief Google Benchmark smoke test for the Atlas benchmark harness.
 *
 * A placeholder case that keeps the benchmark wiring alive until the real cases
 * are (re)written. It confirms that a Google Benchmark target links against the
 * engine and can drive a `System::update()` loop — it is not a representative
 * workload. A tiny fluid of a few moving particles is advected in a unit-box
 * universe with no solver, sink, or source.
 */

#include <atlas/atlas.h>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace {

using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::FluidPositionState;
using atlas::FluidVelocityState;
using atlas::Float3;
using atlas::HostBuffer;
using atlas::System;
using atlas::Universe;

/** @brief A minimal System: eight drifting particles in a unit-box universe. */
System
make_smoke_system() {
    constexpr std::size_t count = 8;

    std::vector<Float3> positions;
    std::vector<Float3> velocities;
    positions.reserve(count);
    velocities.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float f = static_cast<float>(i);
        positions.push_back(Float3(0.1f * f, 0.1f * f, 0.1f * f));
        velocities.push_back(Float3(1.0f, -1.0f, 0.5f));
    }

    FluidHostPtr fluid = Fluid::builder()
                             .with_buffer_size(count)
                             .with_particle_count(count)
                             .make_host_unique();

    const HostBuffer<Float3> host_positions(positions.begin(), positions.end());
    fluid->state<FluidPositionState>()->data()
        = DeviceBuffer<Float3>(host_positions.begin(), host_positions.end());

    const HostBuffer<Float3> host_velocities(velocities.begin(), velocities.end());
    fluid->state<FluidVelocityState>()->data()
        = DeviceBuffer<Float3>(host_velocities.begin(), host_velocities.end());

    Universe::Builder universe = Universe::builder();
    universe.with_lower_corner(Float3(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Float3(1.0f, 1.0f, 1.0f))
        .with_cell_size(1.0f);

    return System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(universe.make_host_unique())
        .with_dt(1.0e-3f)
        .build();
}

/** @brief Steps the smoke system so the harness exercises the update pipeline. */
void
BM_SystemUpdate(benchmark::State& state) {
    // Silence the per-step "no UniverseNumberParticleState" notice: this smoke
    // system runs no solver, so the searcher has nothing to record counts into.
    atlas::Logging::mute();

    System system = make_smoke_system();

    for (auto _ : state) {
        system.update();
        benchmark::DoNotOptimize(system.step());
    }
}

BENCHMARK(BM_SystemUpdate);

}

BENCHMARK_MAIN();
