#include <atlas/atlas.h>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <utility>

namespace {

atlas::System
make_smoke_system() {
    constexpr std::size_t particle_count = 8;
    const std::array<atlas::Float3, particle_count> positions = {
        atlas::Float3(-0.45f, -0.30f, 0.0f),
        atlas::Float3(-0.45f, -0.20f, 0.0f),
        atlas::Float3(-0.45f, -0.10f, 0.0f),
        atlas::Float3(-0.45f, 0.00f, 0.0f),
        atlas::Float3(-0.45f, 0.10f, 0.0f),
        atlas::Float3(-0.45f, 0.20f, 0.0f),
        atlas::Float3(-0.45f, 0.30f, 0.0f),
        atlas::Float3(-0.45f, 0.40f, 0.0f),
    };
    const std::array<atlas::Float3, particle_count> velocities = {
        atlas::Float3(100.0f, 0.0f, 0.0f), atlas::Float3(100.0f, 0.0f, 0.0f),
        atlas::Float3(100.0f, 0.0f, 0.0f), atlas::Float3(100.0f, 0.0f, 0.0f),
        atlas::Float3(100.0f, 0.0f, 0.0f), atlas::Float3(100.0f, 0.0f, 0.0f),
        atlas::Float3(100.0f, 0.0f, 0.0f), atlas::Float3(100.0f, 0.0f, 0.0f),
    };

    auto materials = atlas::MaterialDictionary::builder()
        .with_material(atlas::Material(atlas::Molecule(
            4.65e-26f, 0.0f, 0.0f, 0.0f, 4.17e-10f, 273.0f, 0.74f, 1.0f)))
        .make_host_shared();
    auto fluid = atlas::Fluid::builder()
        .with_buffer_size(particle_count)
        .with_particle_count(particle_count)
        .with_statistical_weight(1.0f)
        .with_materials(materials)
        .make_host_unique();
    const atlas::HostBuffer<atlas::Float3> host_positions(positions.begin(), positions.end());
    const atlas::HostBuffer<atlas::Float3> host_velocities(velocities.begin(), velocities.end());
    fluid->state<atlas::FluidPositionState>()->data() =
        atlas::DeviceBuffer<atlas::Float3>(host_positions.begin(), host_positions.end());
    fluid->state<atlas::FluidVelocityState>()->data() =
        atlas::DeviceBuffer<atlas::Float3>(host_velocities.begin(), host_velocities.end());

    auto universe = atlas::Universe::builder()
        .with_lower_corner(atlas::Float3(-1.0f, -0.75f, -0.5f))
        .with_upper_corner(atlas::Float3(1.5f, 0.75f, 0.5f))
        .with_cell_size(0.25f)
        .make_host_unique();
    const atlas::Geometry cylinder(atlas::Cylinder::builder()
        .with_center(atlas::Float3(0.0f))
        .with_radius(0.2f)
        .with_height(1.0f)
        .with_open(true)
        .build());
    const atlas::Collider collider(atlas::IsothermalCollider::builder()
        .with_unit(atlas::Unit::builder()
            .with_geometry(cylinder)
            .with_sync(atlas::Sync::builder().make_host_shared())
            .build())
        .with_momentum_accommodation_coefficient(1.0f)
        .with_restitution(1.0f)
        .build());

    return atlas::System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_dt(1.0e-4f)
        .with_solver(atlas::DsmcSolver::builder()
            .with_kernel_type(atlas::DsmcKernelType::variable_hard_sphere)
            .with_majorant_sample_pairs(8)
            .with_majorant_exhaustive_limit(5)
            .make_host_shared())
        .with_collider(collider)
        .build();
}

void
BM_CylinderDsmcUpdate(benchmark::State& state) {
    atlas::Logging::mute();
    for (auto _ : state) {
        state.PauseTiming();
        atlas::System system = make_smoke_system();
        state.ResumeTiming();
        system.update();
        benchmark::DoNotOptimize(system.step());
        benchmark::DoNotOptimize(system.fluid()->particle_count());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_CylinderDsmcUpdate);

}

BENCHMARK_MAIN();
