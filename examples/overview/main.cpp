#include <atlas/atlas.h>
#include <vizkit/vizkit.h>

using namespace atlas;

int
main() {
    using sim_t = float;
    using Vec3  = Vector3<sim_t>;

    logger::info() << "\n"
                   << "Starting simulation...";

    const auto box_domain = geometry::Box<sim_t>::builder()
                                .with_lower_corner(Vec3 { -1.0f, -1.0f, -1.0f })
                                .with_upper_corner(Vec3 { 1.0f, 1.0f, 1.0f })
                                .make_host_shared();

    const auto domain = system::Domain<sim_t>::builder()
                            .with_geometry(box_domain)
                            .with_cell_size(0.02f)
                            .make_host_shared();

    const auto nitrogen = MatrialProperties<sim_t>::builder()
                              .with_mass(4.65e-26f)
                              .make_host_shared();

    const auto fluid = system::Fluid<sim_t>::builder()
                           .with_buffer_size(200000)
                           .add_species(nitrogen)
                           .make_host_shared();

    const auto searcher = system::SpatialHashingSearcher<sim_t>::builder()
                              .with_domain(domain)
                              .make_host_shared();

    const auto codec = system::SingleCodec<sim_t>::builder()
                           .with_domain(domain)
                           .with_fluid(fluid)
                           .with_searcher(searcher)
                           .make_host_shared();

    const auto fixed_sync = system::Sync<sim_t>::builder()
                                .make_host_shared();

    const auto collider_geometry = geometry::Box<sim_t>::builder()
                                       .with_lower_corner(Vec3 { -0.5f, -0.5f, -0.5f })
                                       .with_upper_corner(Vec3 { 0.5f, 0.5f, 0.5f })
                                       .make_host_shared();

    const auto collider_unit = system::Unit<sim_t>::builder()
                                   .with_geometry(collider_geometry)
                                   .with_sync(fixed_sync)
                                   .make_host_shared();

    const auto domain_unit = system::Unit<sim_t>::builder()
                                 .with_geometry(box_domain)
                                 .with_sync(fixed_sync)
                                 .make_host_shared();

    const auto source = system::Source<sim_t>::builder()
                            .with_units(HostBuffer<system::Unit<sim_t>> { *collider_unit })
                            .with_fluid(fluid)
                            .with_spawn_types(
                                HostBuffer<system::SpawnType> { system::SpawnType::Volume })
                            .with_spawn_operator(
                                system::SpawnOperator<sim_t>(system::SpawnType::Volume))
                            .with_spacing(20.0f)
                            .with_temperature(300.0f)
                            .make_host_shared();

    const auto sink = system::Sink<sim_t>::builder()
                          .with_units(HostBuffer<system::Unit<sim_t>> { *domain_unit })
                          .with_despawn_types(
                              HostBuffer<system::DespawnType> { system::DespawnType::Volume })
                          .with_despawn_operator(
                              system::DespawnOperator<sim_t>(system::DespawnType::Volume))
                          .with_flip(true)
                          .make_host_shared();

    const auto measure = system::BoltzmanMeasurer<sim_t>::builder()
                             .with_universe(domain)
                             .with_fluid(fluid)
                             .with_searcher(searcher)
                             .with_measure_mode(system::MeasureModeType::All)
                             .make_host_shared();

    const auto interaction = system::ColliderSurfaceInteraction<sim_t>::builder()
                                 .with_restitution(0.95f)
                                 .with_tangential_momentum_accommodation(0.25f)
                                 .with_temperature(300.0f)
                                 .make_host_shared();

    const auto collider = system::Collider<sim_t>::builder()
                              .with_units(HostBuffer<system::Unit<sim_t>> { *collider_unit })
                              .with_surface_interactions(
                                  HostBuffer<system::ColliderSurfaceInteraction<sim_t>> {
                                      *interaction })
                              .make_host_shared();

    const auto dsmc_ntc_solver = system::DsmcNtcSolver<sim_t>::builder()
                                     .with_fluid(fluid)
                                     .make_host_shared();

    const auto solver = system::Orchestrator<sim_t>::builder()
                            .with_solver(dsmc_ntc_solver)
                            .make_host_shared();

    const auto sim_system = system::System<sim_t>::builder()
                                .with_fluid(fluid)
                                .with_dt(0.0025f)
                                .with_domain(domain)
                                .with_codec(codec)
                                .with_source(source)
                                .with_sink(sink)
                                .with_measure(measure)
                                .with_solver(solver)
                                .with_collider(collider)
                                .make_host_shared();

    auto viewer = vizkit::Viewer<sim_t>::builder()
                      .with_system(sim_system)
                      .with_title("ATLAS Overview")
                      .with_size(1440, 900)
                      .build();

    viewer.add_layer(
        vizkit::BoxLayer<sim_t>::builder()
            .with_unit(domain_unit)
            .make_shared());

    viewer.add_layer(
        vizkit::BoxLayer<sim_t>::builder()
            .with_unit(collider_unit)
            .make_shared());

    viewer.add_layer(
        vizkit::ParticleLayer<sim_t>::builder()
            .with_system(sim_system)
            .make_shared());

    return viewer.run();
}
