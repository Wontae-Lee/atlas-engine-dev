#include <atlas/atlas.h>

#include <benchmark/benchmark.h>

namespace {

using namespace atlas;

SystemHostPtr
build_cylinder_continum_system() {
    HostBuffer<MaterialProperties> properties(1);
    HostBuffer<GeneratorHostPtr> generators(1);

    properties[0] = MaterialProperties::builder()
                        .with_type(MaterialType::Molecule)
                        .with_mass(1.0f)
                        .with_molecular_mass(1.0f)
                        .with_rest_density(1000.0f)
                        .with_pressure_coefficient(35.0f)
                        .with_dynamic_viscosity(0.08f)
                        .with_species_id(0)
                        .build();

    generators[0] = JitteringGenerator::builder()
                        .with_base_value(2.4f)
                        .with_jitter_radius(0.25f)
                        .with_seed(17u)
                        .make_host_shared();

    const auto fluid = Fluid::builder()
                           .with_buffer_size(120000)
                           .with_properties(properties)
                           .with_generators(generators)
                           .make_host_shared();

    const auto sync = Sync::builder()
                          .with_rigid_pose(Vector3(0.0f, 0.0f, 0.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
                          .make_host_shared();

    const auto domain_geometry = Box::builder()
                                     .with_lower_corner(Vector3(-5.0f, -2.0f, -1.2f))
                                     .with_upper_corner(Vector3(5.0f, 2.0f, 1.2f))
                                     .build();

    const auto source_geometry = Box::builder()
                                     .with_lower_corner(Vector3(-4.5f, -0.7f, -0.2f))
                                     .with_upper_corner(Vector3(-3.8f, 0.7f, 0.2f))
                                     .build();

    const auto cylinder_geometry = Cylinder::builder()
                                       .with_center(Vector3(0.0f, 0.0f, 0.0f))
                                       .with_radius(0.75f)
                                       .with_height(2.2f)
                                       .build();

    const auto domain_unit = Unit::builder()
                                 .with_geometry(Geometry(domain_geometry))
                                 .with_sync(sync)
                                 .make_host_shared();

    const auto source_unit = Unit::builder()
                                 .with_geometry(Geometry(source_geometry))
                                 .with_sync(sync)
                                 .make_host_shared();

    const auto cylinder_unit = Unit::builder()
                                   .with_geometry(Geometry(cylinder_geometry))
                                   .with_sync(sync)
                                   .make_host_shared();

    const auto universe = Universe::builder()
                              .with_lower_corner(Vector3(-5.0f, -2.0f, -1.2f))
                              .with_upper_corner(Vector3(5.0f, 2.0f, 1.2f))
                              .with_cell_size(0.30f)
                              .with_source_units(HostBuffer<Unit> { *source_unit })
                              .with_sink_units(HostBuffer<Unit> { *domain_unit })
                              .with_collider_units(HostBuffer<Unit> { *cylinder_unit })
                              .make_host_shared();

    const auto searcher = SpatialHashingSearcher::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    const auto sph_solver = SphSolver::builder()
                                .with_universe(universe)
                                .with_fluid(fluid)
                                .with_searcher(searcher)
                                .with_kernel_type(SphKernelType::cubic_spline)
                                .make_host_shared();

    const auto orchestrator = Orchestrator::builder()
                                  .with_universe(universe)
                                  .with_fluid(fluid)
                                  .with_searcher(searcher)
                                  .with_gravity(Vector3(0.0f, 0.0f, 0.0f))
                                  .with_solver(sph_solver)
                                  .make_host_shared();

    const auto source = Source::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_spawn_types(HostBuffer<SpawnType> {
                                SpawnType::Volume,
                            })
                            .with_spawn_operator(Spawn(SpawnType::Volume))
                            .with_spacing(0.16f)
                            .with_temperature(0.0f)
                            .make_host_shared();

    const auto sink = Sink::builder()
                          .with_universe(universe)
                          .with_fluid(fluid)
                          .with_despawn_types(HostBuffer<DespawnType> {
                              DespawnType::Volume,
                          })
                          .with_despawn_operator(Despawn(DespawnType::Volume))
                          .with_flip(true)
                          .make_host_shared();

    const auto collider = Collider::builder()
                              .with_fluid(fluid)
                              .with_universe(universe)
                              .make_host_shared();

    return System::builder()
        .with_fluid(fluid)
        .with_domain(universe)
        .with_source(source)
        .with_sink(sink)
        .with_collider(collider)
        .with_solver(orchestrator)
        .with_dt(1.0e-3f)
        .make_host_shared();
}

void
BM_CylinderContinumSphStep(benchmark::State& state) {
    const auto system = build_cylinder_continum_system();

    for (auto _ : state) {
        system->update();
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["active_particles"] = static_cast<double>(system->fluid()->particle_count());
}

BENCHMARK(BM_CylinderContinumSphStep)->Unit(benchmark::kMillisecond);

}

BENCHMARK_MAIN();
