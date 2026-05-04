#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/dsmc_disjoint_pair_solver.h>

#include <benchmark/benchmark.h>

#include <cstddef>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kDt = T(0.01);
constexpr T kCellSize = T(0.125);

atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::universe::Universe<T>::builder()
        .with_lower_corner(Vec3(0, 0, 0))
        .with_upper_corner(Vec3(1, 1, 1))
        .with_cell_size(kCellSize)
        .make_host_shared();
}

atlas::FluidHostPtr<T>
make_fluid(const std::size_t particle_count) {
    atlas::HostBuffer<atlas::MaterialProperties<T>> properties;
    properties.push_back(
        atlas::MaterialProperties<T>::builder()
            .with_type(atlas::MaterialType::Molecule)
            .with_mass(T(1))
            .with_molecular_mass(T(1))
            .with_collision_diameter(T(1))
            .build());

    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators;
    generators.push_back(nullptr);

    auto fluid = atlas::Fluid<T>::builder()
                     .with_buffer_size(particle_count)
                     .with_properties(properties)
                     .with_generators(generators)
                     .with_statistical_weight(T(1))
                     .make_host_shared();

    fluid->set_particle_count(particle_count);

    auto& positions = fluid->state<atlas::fluid::FluidPositionState<T>>()->data();
    auto& velocities = fluid->state<atlas::fluid::FluidVelocityState<T>>()->data();
    auto& species = fluid->state<atlas::fluid::FluidSpeciesState<T>>()->data();

    for (std::size_t i = 0; i < particle_count; ++i) {
        const auto cell = static_cast<int>(i % 512);
        const int ix = cell % 8;
        const int iy = (cell / 8) % 8;
        const int iz = cell / 64;

        const T jitter_x = T((i * 17) % 97) / T(97) * T(0.75);
        const T jitter_y = T((i * 31) % 89) / T(89) * T(0.75);
        const T jitter_z = T((i * 47) % 83) / T(83) * T(0.75);

        positions[i] = Vec3(
            (T(ix) + T(0.125) + jitter_x) * kCellSize,
            (T(iy) + T(0.125) + jitter_y) * kCellSize,
            (T(iz) + T(0.125) + jitter_z) * kCellSize);

        velocities[i] = Vec3(
            T(1) + T((i * 13) % 29) * T(0.01),
            T(0.5) + T((i * 7) % 23) * T(0.02),
            T(0.25) + T((i * 5) % 19) * T(0.03));

        species[i] = 0u;
    }

    return fluid;
}

atlas::SpatialHashingSearcherHostPtr<T>
make_searcher(const atlas::UniverseHostPtr<T>& universe,
              const atlas::FluidHostPtr<T>& fluid) {
    return atlas::system::SpatialHashingSearcher<T>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

void
BM_DsmcDisjointPairSolverSolve(benchmark::State& state) {
    const auto particle_count = static_cast<std::size_t>(state.range(0));
    const auto universe = make_universe();
    const auto fluid = make_fluid(particle_count);
    const auto searcher = make_searcher(universe, fluid);

    auto solver = atlas::system::DsmcDisjointPairSolver<T>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(atlas::system::DsmcKernelType::hard_sphere)
                      .build();

    solver.solve(kDt);

    for (auto _ : state) {
        solver.solve(kDt);
        benchmark::DoNotOptimize(fluid->state<atlas::fluid::FluidVelocityState<T>>()->data().data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(particle_count));
    state.counters["cells"] = universe->number_of_cells();
}

BENCHMARK(BM_DsmcDisjointPairSolverSolve)
    ->Name("DsmcDisjointPairSolver/solve")
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536)
    ->Unit(benchmark::kMillisecond);

} // namespace
