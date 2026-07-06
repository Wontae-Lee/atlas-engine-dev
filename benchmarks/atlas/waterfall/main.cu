#include <atlas/atlas.h>

#include <benchmark/benchmark.h>

#include <utility>

namespace {

using namespace atlas;

SystemHostPtr
build_waterfall_system() {
    HostBuffer<MaterialProperties> material_properties(1);
    HostBuffer<GeneratorHostPtr> generators(1);

    material_properties[0] = MaterialProperties::builder()
                                 .with_type(MaterialType::Molecule)
                                 .with_mass(1.0f)
                                 .with_molecular_mass(1.0f)
                                 .with_rest_density(1.0f)
                                 .with_pressure_coefficient(6.5f)
                                 .with_dynamic_viscosity(0.035f)
                                 .with_species_id(0)
                                 .build();

    generators[0] = JitteringGenerator::builder()
                        .with_base_value(-2.8f)
                        .with_jitter_radius(0.35f)
                        .with_seed(7u)
                        .make_host_shared();

    const auto fluid = Fluid::builder()
                           .with_buffer_size(90000)
                           .with_properties(material_properties)
                           .with_generators(generators)
                           .make_host_shared();

    const auto domain_geometry = Box::builder()
                                     .with_lower_corner(Vector3(-40.0f, -20.0f, -10.0f))
                                     .with_upper_corner(Vector3(6.0f, 20.0f, 10.0f))
                                     .build();

    const auto source_geometry = Box::builder()
                                     .with_lower_corner(Vector3(5.5f, -14.5f, 3.2f))
                                     .with_upper_corner(Vector3(6.0f, 14.5f, 8.4f))
                                     .build();

    const float   half_extent = 4.5f;
    const Vector3 v0(half_extent, half_extent, half_extent);
    const Vector3 v1(-half_extent, -half_extent, half_extent);
    const Vector3 v2(-half_extent, half_extent, -half_extent);
    const Vector3 v3(half_extent, -half_extent, -half_extent);

    HostBuffer<TriangleContainer4> triangles(4);
    auto write_face = [&](const std::size_t face_index,
                          const Vector3&    a,
                          const Vector3&    b,
                          const Vector3&    c,
                          const Vector3&    opposite_vertex) {
        auto& triangle = triangles[face_index];
        triangle.a()   = a;
        triangle.b()   = b;
        triangle.c()   = c;

        const Vector3 face_center = (a + b + c) / static_cast<float>(3);
        const Vector3 face_normal = cross(triangle.b() - triangle.a(), triangle.c() - triangle.a());

        if (dot(face_normal, opposite_vertex - face_center) > static_cast<float>(0)) {
            std::swap(triangle.b(), triangle.c());
        }
    };

    write_face(0, v0, v1, v2, v3);
    write_face(1, v0, v3, v1, v2);
    write_face(2, v0, v2, v3, v1);
    write_face(3, v1, v3, v2, v0);

    const auto tetrahedron_geometry = TriangleMesh::builder()
                                          .with_triangles(std::move(triangles))
                                          .build();

    const auto sync = Sync::builder()
                          .with_rigid_pose(Vector3(0, 0, 0), Quaternion(1, 0, 0, 0))
                          .make_host_shared();

    const auto domain_unit = Unit::builder()
                                 .with_geometry(Geometry(domain_geometry))
                                 .with_sync(sync)
                                 .make_host_shared();

    const auto source_unit = Unit::builder()
                                 .with_geometry(Geometry(source_geometry))
                                 .with_sync(sync)
                                 .make_host_shared();

    const auto tetrahedron_unit = Unit::builder()
                                      .with_geometry(tetrahedron_geometry.make_device_geometry_view())
                                      .with_sync(sync)
                                      .with_angular_velocity(Vector3(0.35f, 0.55f, 0.90f))
                                      .make_host_shared();

    const auto universe = Universe::builder()
                              .with_lower_corner(Vector3(-40.0f, -20.0f, -10.0f))
                              .with_upper_corner(Vector3(6.0f, 20.0f, 10.0f))
                              .with_cell_size(0.32f)
                              .with_source_units(HostBuffer<Unit> { *source_unit })
                              .with_sink_units(HostBuffer<Unit> { *domain_unit })
                              .with_collider_units(HostBuffer<Unit> { *tetrahedron_unit })
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
                                  .with_gravity(Vector3(0.0f, 0.0f, -9.81f))
                                  .with_solver(sph_solver)
                                  .make_host_shared();

    const auto source = Source::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_spawn_types(HostBuffer<SpawnType> {
                                SpawnType::Volume,
                            })
                            .with_spawn_operator(Spawn(SpawnType::Volume))
                            .with_spacing(1.0f)
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
        .with_dt(0.004f)
        .make_host_shared();
}

void
BM_WaterfallSphStep(benchmark::State& state) {
    const auto system = build_waterfall_system();

    for (auto _ : state) {
        system->update();
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["active_particles"] = static_cast<double>(system->fluid()->particle_count());
}

BENCHMARK(BM_WaterfallSphStep)->Unit(benchmark::kMillisecond);

}

BENCHMARK_MAIN();
