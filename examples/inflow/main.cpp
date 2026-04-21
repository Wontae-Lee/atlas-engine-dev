#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <iostream>

namespace {

using T    = float;
using Vec3 = atlas::Vector3<T>;

namespace config {

    // Gas and time-integration parameters used throughout the example.
    //
    // This scene models nitrogen flowing inside a finite cylinder.
    // Particles are injected from a circular source near the +z cap with no
    // prescribed bulk drift and then evolve under:
    // - DSMC collisions between particles
    // - collisions against the cylinder wall volume
    // - sink removal outside the cylinder's axis-aligned bounds
    constexpr T kTemperature           = 300.0f;
    constexpr T kDt                    = 2.5e-5f;
    constexpr std::size_t kBufferSize  = 200000;
    constexpr T kNitrogenMolecularMass = 4.651734e-26f;
    constexpr T kNitrogenDiameter      = 4.17e-10f;

    // Spatial-discretization and emission-density controls.
    //
    // kCellSize controls the background Cartesian grid used by the spatial hashing
    // searcher and DSMC collision solver.
    //
    // kSourceSpacing controls how densely the source box is sampled with candidate
    // emission points. Smaller values increase the number of emitted particles per
    // update, which makes the flow field visually denser but also more expensive.
    constexpr T kCellSize      = 0.25f;
    constexpr T kSourceSpacing = 0.15f;

    // Viewer-side presentation parameters.
    constexpr int kViewerWidth    = 1440;
    constexpr int kViewerHeight   = 900;
    constexpr int kCylinderSlices = 72;

    // Number of updates to execute when the example is built without Vizkit.
    constexpr int kHeadlessSteps = 1000;

    // Pipe geometry.
    const Vec3 kCylinderCenter(0, 0, 0);
    constexpr T kCylinderRadius = 1.0f;
    constexpr T kCylinderHeight = 8.0f;

    // Circular source placed near the +z side of the cylinder.
    const Vec3 kSourceCenter(0.0f, 0.0f, 3.6f);
    const Vec3 kSourceNormal(0.0f, 0.0f, 1.0f);
    constexpr T kSourceRadius = 1.f;

    // The source emits a thermalized fluid with no prescribed bulk drift.
    const Vec3 kBulkVelocity(0.0f, 0.0f, 0.0f);

    // Surface-interaction settings for the cylinder.
    //
    // This example uses a fully diffuse reflection model:
    // - cosine-weighted hemisphere sampling
    // - TMAC = 1, so the diffuse branch is always selected
    // - restitution = 1, so reflected particles keep their pre-collision speed
    constexpr atlas::system::DiffuseSampling kDiffuseSampling
        = atlas::system::DiffuseSampling::CosineWeighted;
    constexpr T kRestitution                     = 1.0f;
    constexpr T kTangentialMomentumAccommodation = 1.0f;

    // Particle-particle collisions use the hard-sphere DSMC kernel.
    constexpr atlas::system::DsmcKernelType kDsmcKernelType
        = atlas::system::DsmcKernelType::hard_sphere;

    // Measure macroscopic cell fields from the evolving particle distribution.
    //
    // `Field` keeps the measurement on the universe grid, which is sufficient for
    // this example because the result is used as a cell-wise diagnostic rather than
    // being written back into per-particle temperature state.
    constexpr atlas::MeasureModeType kMeasureMode = atlas::MeasureModeType::Field;

    // Simple visual styling for the particle cloud and window title.
    const atlas::Vector4<T> kParticleColor(0.10f, 0.74f, 0.92f, 0.80f);
    const atlas::Vector4<T> kCylinderColor(0.92f, 0.96f, 0.98f, 0.28f);
    constexpr const char* kViewerTitle = "Atlas DSMC Nitrogen In-Cylinder Flow";

} // namespace config

atlas::UnitHostPtr<T>
make_unit(const atlas::GeometryHostPtr<T>& geometry) {
    // Units in this example are static world-space objects.
    //
    // The geometry defines the analytic shape, while Sync supplies the rigid
    // transform used by collider queries and Vizkit geometry layers.
    const auto sync = atlas::Sync<T>::builder()
                          .with_rigid_pose(Vec3(0, 0, 0), atlas::Quaternion<T>(1, 0, 0, 0))
                          .make_host_shared();

    return atlas::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .make_host_shared();
}

atlas::FluidHostPtr<T>
make_fluid() {
    // Atlas fluids store:
    // - species/material properties
    // - host-configured generator models
    // - fixed-capacity particle state buffers
    //
    // This example uses a single nitrogen species, so both the material table
    // and generator table contain exactly one entry.
    atlas::HostBuffer<atlas::MatrialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    // Material properties required by the DSMC solver.
    //
    // The hard-sphere kernel reads:
    // - molecular_mass
    // - collision_diameter
    //
    // species_id is also installed so the runtime has an explicit species label.
    properties[0] = atlas::MatrialProperties<T>::builder()
                        .with_type(atlas::MaterialType::Molecule)
                        .with_mass(config::kNitrogenMolecularMass)
                        .with_molecular_mass(config::kNitrogenMolecularMass)
                        .with_species_id(0)
                        .with_collision_diameter(config::kNitrogenDiameter)
                        .build();

    // Source emission uses a Maxwell-Boltzmann velocity generator with zero
    // bulk drift.
    generators[0] = atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
                        .with_temperature(config::kTemperature)
                        .with_molecular_mass(config::kNitrogenMolecularMass)
                        .with_bulk_velocity(config::kBulkVelocity)
                        .with_seed(42u)
                        .make_host_shared();

    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(config::kBufferSize)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_source_geometry() {
    // A circular source disk near the +z cap injects particles from the upper
    // side of the cylinder.
    return atlas::geometry::Circle<T>::builder()
        .with_center(config::kSourceCenter)
        .with_normal(config::kSourceNormal)
        .with_radius(config::kSourceRadius)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_cylinder_geometry() {
    // The obstacle is a centered finite cylinder aligned with the z-axis.
    return atlas::geometry::Cylinder<T>::builder()
        .with_center(config::kCylinderCenter)
        .with_radius(config::kCylinderRadius)
        .with_height(config::kCylinderHeight)
        .with_open(true)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_bound_geometry(const atlas::GeometryHostPtr<T>& geometry) {
    const auto bounds = geometry->bound();
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(bounds.lower_corner)
        .with_upper_corner(bounds.upper_corner)
        .make_host_shared();
}

atlas::system::ColliderSurfaceInteraction<T>
make_collider_interaction() {
    // Construct the cylinder reflection law used by the collider.
    //
    // Diffuse reflection randomizes only the outgoing direction; the magnitude
    // is still controlled by restitution inside ColliderSurfaceInteraction.
    return atlas::system::ColliderSurfaceInteraction<T>::builder()
        .with_diffuse_sampling(config::kDiffuseSampling)
        .with_restitution(config::kRestitution)
        .with_tangential_momentum_accommodation(config::kTangentialMomentumAccommodation)
        .with_temperature(config::kTemperature)
        .build();
}

} // namespace

int
main() {
    // Create the particle container first because almost every other runtime
    // subsystem depends on it.
    const auto fluid = make_fluid();

    // Universe defines the simulation extents and regular cell grid.
    //
    // The spatial hashing searcher and DSMC solver both use this grid.
    const auto cylinder_geometry = make_cylinder_geometry();
    const auto domain_geometry   = make_bound_geometry(cylinder_geometry);
    const auto source_geometry   = make_source_geometry();
    const auto bounds            = domain_geometry->bound();

    const auto universe = atlas::Universe<T>::builder()
                              .with_lower_corner(bounds.lower_corner)
                              .with_upper_corner(bounds.upper_corner)
                              .with_cell_size(config::kCellSize)
                              .make_host_shared();

    // SpatialHashingSearcher maps active particles into grid cells each step so
    // the DSMC solver can find local collision neighborhoods efficiently.
    const auto searcher = atlas::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    // Build the DSMC solver that handles particle-particle collisions.
    const auto dsmc_solver = atlas::DsmcNtcSolver<T>::builder()
                                 .with_universe(universe)
                                 .with_fluid(fluid)
                                 .with_searcher(searcher)
                                 .with_kernel_type(config::kDsmcKernelType)
                                 .make_host_shared();

    // BoltzmanMeasurer computes macroscopic cell fields such as bulk velocity
    // and temperature from the current particle distribution after search.
    const auto measurer = atlas::BoltzmanMeasurer<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_measure_mode(config::kMeasureMode)
                              .make_host_shared();

    // The orchestrator owns the "search -> measure -> solve" runtime sequence.
    const auto orchestrator = atlas::Orchestrator<T>::builder()
                                  .with_universe(universe)
                                  .with_fluid(fluid)
                                  .with_searcher(searcher)
                                  .with_measurer(measurer)
                                  .with_solver(dsmc_solver)
                                  .make_host_shared();

    // Wrap each geometry in a Unit so the same objects can be used by runtime
    // systems and by Vizkit layers.
    const auto domain_unit   = make_unit(domain_geometry);
    const auto source_unit   = make_unit(source_geometry);
    const auto cylinder_unit = make_unit(cylinder_geometry);

    // Source emits surface samples from the circular source at thermal equilibrium.
    const auto source = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { *source_unit })
                            .with_fluid(fluid)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                                atlas::fluid::SpawnType::Surface,
                            })
                            .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Surface))
                            .with_spacing(config::kSourceSpacing)
                            .with_temperature(config::kTemperature)
                            .make_host_shared();

    // Sink removes particles that leave the cylinder's bounding box.
    const auto sink = atlas::fluid::Sink<T>::builder()
                          .with_units(atlas::HostBuffer<atlas::Unit<T>> { *domain_unit })
                          .with_fluid(fluid)
                          .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {
                              atlas::fluid::DespawnType::Volume,
                          })
                          .with_despawn_operator(atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Volume))
                          .with_flip(true)
                          .make_host_shared();

    // Collider applies diffuse wall reflection on the inside of the cylinder.
    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *cylinder_unit })
                              .with_surface_interactions(
                                  atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> {
                                      make_collider_interaction(),
                                  })
                              .with_flip(true)
                              .make_host_shared();

    // Assemble the top-level runtime system.
    //
    // Per-frame update order is:
    // 1. source emits particles
    // 2. orchestrator runs DSMC collisions
    // 3. collider resolves in-cylinder wall hits
    // 4. sink removes escaped particles
    const auto system = atlas::System<T>::builder()
                            .with_fluid(fluid)
                            .with_domain(universe)
                            .with_source(source)
                            .with_sink(sink)
                            .with_collider(collider)
                            .with_solver(orchestrator)
                            .with_dt(config::kDt)
                            .make_host_shared();

#ifdef ATLAS_ENABLE_VIZKIT
    // When Vizkit is enabled, run the full interactive viewer.
    atlas::vizkit::Viewer<T> viewer = atlas::vizkit::Viewer<T>::builder()
                                          .with_system(system)
                                          .with_title(config::kViewerTitle)
                                          .with_size(config::kViewerWidth, config::kViewerHeight)
                                          .build();

    // Frame the initial camera around the full domain.
    viewer.camera().fit_bounds(bounds.lower_corner, bounds.upper_corner);

    // Render the live particle cloud.
    viewer.add_layer(
        atlas::vizkit::ParticleLayer<T>::builder()
            .with_system(system)
            .with_color(config::kParticleColor)
            .make_shared());

    // Render the domain bounds as a wireframe reference.
    viewer.add_layer(
        atlas::vizkit::BoxLayer<T>::builder()
            .with_unit(domain_unit)
            .make_shared());

    // Render the obstacle cylinder as a semi-transparent wireframe reference.
    const auto cylinder_layer = atlas::vizkit::CylinderLayer<T>::builder()
                                    .with_unit(cylinder_unit)
                                    .with_slices(config::kCylinderSlices)
                                    .make_shared();
    cylinder_layer->set_color(config::kCylinderColor);
    viewer.add_layer(cylinder_layer);

    std::cout
        << "Nitrogen DSMC example: 300 K, zero bulk drift, circular source near +z, flow inside a finite cylinder.\n";

    return viewer.run();
#else
    // Headless fallback for non-Vizkit builds.
    //
    // This keeps the example runnable in environments without OpenGL/GLFW while
    // still exercising the same simulation pipeline.
    for (int step = 0; step < config::kHeadlessSteps; ++step) {
        system->update();
    }

    std::cout
        << "Nitrogen DSMC example ran headlessly for " << config::kHeadlessSteps << " steps.\n"
        << "Active particles: " << fluid->particle_count() << '\n';
    return 0;
#endif
}
