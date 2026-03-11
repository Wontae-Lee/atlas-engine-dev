#include <atlas/atlas.h>
using namespace atlas;

auto box_layer    = std::make_shared<BoxLayer<float>>(Vector3F { -10.0f, -10.0f, -10.0f }, Vector3F { 10.0f, 10.0f, 10.0f });
auto sphere_layer = std::make_shared<SphereLayer<float>>(Vector3<float> { 0, 0, 0 }, 3.f);

Viewer<float> viewer(1280, 720, "Atlas CUDA Particles");
viewer.add_layer(box_layer);
viewer.add_layer(sphere_layer);
viewer.run();


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
    const auto searcher = system::SpatialHashingSearcher<sim_t>::builder()
                              .with_domain(domain)
                              .with_range(system::NeighborSearchRange::single)
                              .make_host_shared();

    // ------------------------------------------------------------
    // Codec for encoding/decoding particle states inside the domain
    //
    // The codec defines how particle data is packed, stored,
    // and interpreted during simulation.
    // ------------------------------------------------------------
    const auto codec = system::SingleCodec<sim_t>::builder()
                           .with_domain(domain)
                           .make_host_shared();

    // ------------------------------------------------------------
    // Define a fluidic particle species (e.g., Nitrogen gas)
    //
    // The molecular mass is specified in kilograms.
    // ------------------------------------------------------------
    const auto nitrogen = FluidicParticle<sim_t>::builder()
                              .with_molecular_mass(4.65e-26)
                              .make_host_shared();

    // ------------------------------------------------------------
    // Create a fluid composed of one or more particle species
    //
    // Fluids manage collections of particle definitions and
    // associated physical properties.
    // ------------------------------------------------------------
    const auto fluid = system::Fluid<sim_t>::builder()
                           .add_particle(nitrogen)
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

    (void)searcher;
    (void)codec;
    (void)fluid;

    return EXIT_SUCCESS;
}
