#include <atlas/atlas.h>
using namespace atlas;

int
main() {
    using sim_t = float;

    // ------------------------------------------------------------
    // Define the global simulation domain as an axis-aligned box
    // spanning from (-1, -1, -1) to (1, 1, 1)
    // ------------------------------------------------------------
    const auto box_domain = geometry::Box<sim_t>::builder()
                                .with_lower_corner(Vector3<sim_t> { -1.0f, -1.0f, -1.0f })
                                .with_upper_corner(Vector3<sim_t> { 1.0f, 1.0f, 1.0f })
                                .make_host_shared();

    atlas::logger::info() << "\n"
                          << "Starting simulation...";

    // ------------------------------------------------------------
    // Create the simulation domain and spatial discretization
    //
    // The domain defines:
    // - global coordinate bounds
    // - uniform cell size used for spatial partitioning
    // ------------------------------------------------------------
    const auto domain = system::Domain<sim_t>::builder()
                            .with_geometry(box_domain)
                            .with_cell_size(0.02f)
                            .make_host_shared();

    // ------------------------------------------------------------
    // Construct the neighbor-search structure
    //
    // SpatialHashingSearcher is responsible for:
    // - mapping positions to grid cells
    // - efficiently finding nearby particles or units
    // ------------------------------------------------------------
    const auto codec = system::SingleCodec<sim_t>::builder()
                           .with_domain(domain)
                           .make_host_shared();

    // ------------------------------------------------------------
    // Define a fluidic particle species (e.g., Nitrogen gas)
    //
    // The molecular mass is specified in kilograms.
    // ------------------------------------------------------------
    const auto nitrogen = MatrialProperties<sim_t>::builder()
                              .with_mass(4.65e-26f)
                              .make_host_shared();

    // ------------------------------------------------------------
    // Create a fluid composed of one or more particle species
    //
    // Fluids manage collections of particle definitions and
    // associated physical properties.
    // ------------------------------------------------------------
    const auto fluid = system::Fluid<sim_t>::builder()
                           .with_buffer_size(200000)
                           .add_species(nitrogen)
                           .make_host_shared();

    // ------------------------------------------------------------
    // Define a geometric unit to be inserted into the simulation
    //
    // This unit represents a smaller box located at the center
    // of the domain, acting as a solid or interaction object.
    // ------------------------------------------------------------
    const auto box_unit = geometry::Box<sim_t>::builder()
                              .with_lower_corner(Vector3<sim_t> { -0.5f, -0.5f, -0.5f })
                              .with_upper_corner(Vector3<sim_t> { 0.5f, 0.5f, 0.5f })
                              .make_host_shared();

    // ------------------------------------------------------------
    // Build a synchronization policy for the unit
    //
    // Identity sync implies:
    // - the unit remains stationary
    // - no time-dependent transformation is applied
    // ------------------------------------------------------------
    const auto fixed_sync = system::Sync<sim_t>::builder()
                                .make_host_shared();

    // ------------------------------------------------------------
    // Assemble the simulation unit by combining geometry and sync
    //
    // Unit<T> represents:
    // - a geometric object
    // - its spatial transformation and motion policy
    // ------------------------------------------------------------
    const auto unit = system::Unit<sim_t>::builder()
                          .with_geometry(box_unit)
                          .with_sync(fixed_sync)
                          .make_host_shared();

    const auto domain_unit = system::Unit<sim_t>::builder()
                                 .with_geometry(box_domain)
                                 .with_sync(fixed_sync)
                                 .make_host_shared();

    const auto source = system::Source<sim_t>::builder()
                            .with_unit(*unit)
                            .with_fluid(fluid)
                            .with_spawn_type(system::SpawnType::Volume)
                            .with_spacing(20.f)
                            .with_temperature(300.0f)
                            .make_host_shared();

    const auto sink = system::Sink<sim_t>::builder()
                          .with_unit(*domain_unit)
                          .with_despawn_type(system::DespawnType::Volume)
                          .with_flip(true)
                          .make_host_shared();

    const auto measure = system::VarianceThermometer<sim_t>::builder()
                             .make_host_shared();

    const auto interaction = system::ColliderSurfaceInteraction<sim_t>::builder()
                                 .with_restitution(0.95f)
                                 .with_tangential_momentum_accommodation(0.25f)
                                 .with_temperature(300.0f)
                                 .make_host_shared();

    const auto collider = system::Collider<sim_t>::builder()
                              .with_unit(unit)
                              .with_surface_interaction(interaction)
                              .make_host_shared();

    const auto solver = atlas::make_host_shared<Solver<sim_t>>();

    const auto sim_system = system::System<sim_t>::builder()
                                .with_fluid(fluid)
                                .with_dt(0.01f)
                                .with_domain(domain)
                                .with_codec(codec)
                                .with_source(source)
                                .with_sink(sink)
                                .with_measure(measure)
                                .with_solver(solver)
                                .with_collider(collider)
                                .make_host_shared();

    sim_system->update();

    return EXIT_SUCCESS;
}
