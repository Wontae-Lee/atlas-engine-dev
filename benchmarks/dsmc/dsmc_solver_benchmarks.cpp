#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

#include <benchmark/benchmark.h>

namespace {

using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::GeneratorHostPtr;
using atlas::HostBuffer;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::SpatialHashingSearcherHostPtr;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3F;
using atlas::system::DsmcApplyMode;
using atlas::system::DsmcKernelType;
using atlas::system::DsmcMajorantMode;
using atlas::system::DsmcSolver;
using atlas::system::SpatialHashingSearcher;

using FluidPositionState = atlas::fluid::FluidPositionState<float>;
using FluidSpeciesState  = atlas::fluid::FluidSpeciesState<float>;
using FluidVelocityState = atlas::fluid::FluidVelocityState<float>;

UniverseHostPtr<float>
make_benchmark_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Vector3F(1.0f, 1.0f, 1.0f))
        .with_cell_size(1.0f)
        .make_host_shared();
}

FluidHostPtr<float>
make_benchmark_fluid(const int particle_count) {
    HostBuffer<MaterialProperties<float>> properties;
    properties.push_back(
        MaterialProperties<float>::builder()
            .with_type(MaterialType::Molecule)
            .with_mass(1.0f)
            .with_molecular_mass(1.0f)
            .with_reference_diameter(1.0f)
            .build());

    HostBuffer<GeneratorHostPtr<float>> generators;
    generators.push_back(nullptr);

    auto fluid = Fluid<float>::builder()
                     .with_buffer_size(static_cast<std::size_t>(particle_count))
                     .with_properties(properties)
                     .with_generators(generators)
                     .make_host_shared();

    fluid->set_particle_count(static_cast<std::size_t>(particle_count));

    auto& positions = fluid->state<FluidPositionState>()->data();
    auto& velocities = fluid->state<FluidVelocityState>()->data();
    auto& species = fluid->state<FluidSpeciesState>()->data();

    for (int i = 0; i < particle_count; ++i) {
        const float u = static_cast<float>((i * 37) % 997) / 997.0f;
        const float v = static_cast<float>((i * 73) % 991) / 991.0f;
        const float w = static_cast<float>((i * 19) % 983) / 983.0f;
        positions[static_cast<std::size_t>(i)] = Vector3F(u, v, w);
        velocities[static_cast<std::size_t>(i)] = Vector3F(
            1.0f + static_cast<float>(i % 7) * 0.1f,
            0.5f - static_cast<float>(i % 5) * 0.05f,
            -0.25f + static_cast<float>(i % 3) * 0.1f);
        species[static_cast<std::size_t>(i)] = 0u;
    }

    return fluid;
}

SpatialHashingSearcherHostPtr<float>
make_benchmark_searcher(const UniverseHostPtr<float>& universe,
                        const FluidHostPtr<float>& fluid) {
    return SpatialHashingSearcher<float>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

void
run_dsmc_benchmark(benchmark::State& state,
                   const DsmcApplyMode apply_mode,
                   const DsmcMajorantMode majorant_mode,
                   const bool use_snapshot) {
    const int particle_count = static_cast<int>(state.range(0));
    const auto universe = make_benchmark_universe();
    const auto fluid = make_benchmark_fluid(particle_count);
    const auto searcher = make_benchmark_searcher(universe, fluid);

    auto solver = DsmcSolver<float>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(DsmcKernelType::hard_sphere)
                      .with_apply_mode(apply_mode)
                      .with_majorant_mode(majorant_mode)
                      .with_majorant_sample_count(64)
                      .with_majorant_safety_factor(majorant_mode == DsmcMajorantMode::exact_all_pairs ? 1.0f : 1.05f)
                      .with_majorant_decay_factor(majorant_mode == DsmcMajorantMode::exact_all_pairs ? 1.0f : 0.99f)
                      .with_pre_collision_snapshot(use_snapshot)
                      .build();

    for (auto _ : state) {
        solver.solve(1.0e-3f);
        benchmark::DoNotOptimize(fluid->state<FluidVelocityState>()->data().data());
        benchmark::DoNotOptimize(universe->state<atlas::universe::UniverseCollisionCountState<int>>()->data().data());
    }

    state.SetItemsProcessed(state.iterations() * particle_count);
}

void
BM_DsmcCellSequentialExact(benchmark::State& state) {
    run_dsmc_benchmark(
        state,
        DsmcApplyMode::cell_sequential,
        DsmcMajorantMode::exact_all_pairs,
        false);
}

void
BM_DsmcFlattenedAtomicExact(benchmark::State& state) {
    run_dsmc_benchmark(
        state,
        DsmcApplyMode::flattened_atomic,
        DsmcMajorantMode::exact_all_pairs,
        false);
}

void
BM_DsmcFlattenedAtomicSampled(benchmark::State& state) {
    run_dsmc_benchmark(
        state,
        DsmcApplyMode::flattened_atomic,
        DsmcMajorantMode::sampled,
        false);
}

void
BM_DsmcFlattenedAtomicSampledSnapshot(benchmark::State& state) {
    run_dsmc_benchmark(
        state,
        DsmcApplyMode::flattened_atomic,
        DsmcMajorantMode::sampled,
        true);
}

} // namespace

BENCHMARK(BM_DsmcCellSequentialExact)->RangeMultiplier(2)->Range(32, 512);
BENCHMARK(BM_DsmcFlattenedAtomicExact)->RangeMultiplier(2)->Range(32, 512);
BENCHMARK(BM_DsmcFlattenedAtomicSampled)->RangeMultiplier(2)->Range(32, 512);
BENCHMARK(BM_DsmcFlattenedAtomicSampledSnapshot)->RangeMultiplier(2)->Range(32, 512);
