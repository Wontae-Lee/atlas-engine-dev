#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <iostream>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

namespace config {

// Gas and time-integration parameters used throughout the example.
//
// This scene models nitrogen at room temperature with no prescribed bulk drift.
// Particles are injected thermally and then evolve under:
// - DSMC collisions between particles
// - collisions against the central cylinder
// - sink removal outside the simulation domain
constexpr T kTemperature = 300.0f;
constexpr T kDt = 2.5e-5f;
constexpr std::size_t kBufferSize = 200000;
constexpr T kNitrogenMolecularMass = 4.651734e-26f;
constexpr T kNitrogenDiameter = 4.17e-10f;

// Spatial-discretization and emission-density controls.
//
// kCellSize controls the background Cartesian grid used by the spatial hashing
// searcher and DSMC collision solver.
//
// kSourceSpacing controls how densely the source box is sampled with candidate
// emission points. Smaller values increase the number of emitted particles per
// update, which makes the flow field visually denser but also more expensive.
constexpr T kCellSize = 0.25f;
constexpr T kSourceSpacing = 0.18f;

// Viewer-side presentation parameters.
constexpr int kViewerWidth = 1440;
constexpr int kViewerHeight = 900;
constexpr int kCylinderSlices = 72;

// Number of updates to execute when the example is built without Vizkit.
constexpr int kHeadlessSteps = 1000;

// Outer rectangular domain.
//
// The domain is intentionally shallow in y relative to x so the flow reads as
// a channel around the cylinder, but tall enough in z that particles can
// redistribute after diffuse reflection instead of immediately clipping out.
const Vec3 kDomainMin(-6.0f, -2.5f, -1.2f);
const Vec3 kDomainMax(6.0f, 2.5f, 1.2f);

// Source volume placed inside the left side of the domain.
//
// The source is a finite box rather than an infinitesimally thin inlet plane so
// particles are visible immediately after spawning and the source is robust even
// when the camera is zoomed in.
const Vec3 kSourceMin(-5.1f, -0.9f, -0.18f);
const Vec3 kSourceMax(-4.0f, 0.9f, 0.18f);

// Central obstacle geometry.
const Vec3 kCylinderCenter(0, 0, 0);
constexpr T kCylinderRadius = 0.9f;
constexpr T kCylinderHeight = 2.2f;

// Surface-interaction settings for the cylinder.
//
// This example uses a fully diffuse reflection model:
// - cosine-weighted hemisphere sampling
// - TMAC = 1, so the diffuse branch is always selected
// - restitution = 1, so reflected particles keep their pre-collision speed
constexpr atlas::system::DiffuseSampling kDiffuseSampling
    = atlas::system::DiffuseSampling::CosineWeighted;
constexpr T kRestitution = 1.0f;
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
constexpr const char* kViewerTitle = "Atlas DSMC Nitrogen Cylinder";

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

    // Source emission uses a Maxwell-Boltzmann velocity generator.
    //
    // bulk_velocity is set to zero so the flow has no imposed drift; all motion
    // comes from thermal emission plus subsequent DSMC/collider interactions.
    generators[0] = atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
                        .with_temperature(config::kTemperature)
                        .with_molecular_mass(config::kNitrogenMolecularMass)
                        .with_bulk_velocity(Vec3(0, 0, 0))
                        .with_seed(42u)
                        .make_host_shared();

    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(config::kBufferSize)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_domain_geometry() {
    // The domain geometry serves two roles:
    // 1. a visual wireframe box in Vizkit
    // 2. the sink's reference volume for deciding whether particles remain
    //    inside the valid simulation region
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(config::kDomainMin)
        .with_upper_corner(config::kDomainMax)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_source_geometry() {
    // A compact source box near the left boundary continuously injects thermal
    // particles into the channel.
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(config::kSourceMin)
        .with_upper_corner(config::kSourceMax)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_cylinder_geometry() {
    // The obstacle is a centered finite cylinder aligned with the z-axis.
    return atlas::geometry::Cylinder<T>::builder()
        .with_center(config::kCylinderCenter)
        .with_radius(config::kCylinderRadius)
        .with_height(config::kCylinderHeight)
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
    const auto universe = atlas::Universe<T>::builder()
                              .with_lower_corner(config::kDomainMin)
                              .with_upper_corner(config::kDomainMax)
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

    // Build the three scene geometries:
    // - the outer domain box
    // - the left-side source volume
    // - the centered cylinder collider
    const auto domain_geometry = make_domain_geometry();
    const auto source_geometry = make_source_geometry();
    const auto cylinder_geometry = make_cylinder_geometry();

    // Wrap each geometry in a Unit so the same objects can be used by runtime
    // systems and by Vizkit layers.
    const auto domain_unit = make_unit(domain_geometry);
    const auto source_unit = make_unit(source_geometry);
    const auto cylinder_unit = make_unit(cylinder_geometry);

    // Source emits volume samples from the source box at thermal equilibrium.
    //
    // There is no imposed streamwise bulk velocity; visible advection emerges
    // from repeated thermal injection plus collisions and obstacle scattering.
    const auto source = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { *source_unit })
                            .with_fluid(fluid)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                                atlas::fluid::SpawnType::Volume,
                            })
                            .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Volume))
                            .with_spacing(config::kSourceSpacing)
                            .with_temperature(config::kTemperature)
                            .make_host_shared();

    // Sink removes particles outside the simulation domain.
    //
    // The sink tests against the domain volume and uses flip=true:
    // - "inside domain"  -> keep particle
    // - "outside domain" -> remove particle
    const auto sink = atlas::fluid::Sink<T>::builder()
                          .with_units(atlas::HostBuffer<atlas::Unit<T>> { *domain_unit })
                          .with_fluid(fluid)
                          .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {
                              atlas::fluid::DespawnType::Volume,
                          })
                          .with_despawn_operator(atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Volume))
                          .with_flip(true)
                          .make_host_shared();

    // Collider applies diffuse wall reflection against the cylinder.
    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *cylinder_unit })
                              .with_surface_interactions(
                                  atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> {
                                      make_collider_interaction(),
                                  })
                              .make_host_shared();

    // Assemble the top-level runtime system.
    //
    // Per-frame update order is:
    // 1. source emits particles
    // 2. orchestrator runs DSMC collisions
    // 3. collider resolves obstacle hits
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
    viewer.camera().fit_bounds(config::kDomainMin, config::kDomainMax);

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

    // Render the obstacle cylinder as a wireframe reference.
    viewer.add_layer(
        atlas::vizkit::CylinderLayer<T>::builder()
            .with_unit(cylinder_unit)
            .with_slices(config::kCylinderSlices)
            .make_shared());

    std::cout
        << "Nitrogen DSMC example: 300 K, zero bulk velocity, inlet-only source, outlet sinks, centered cylinder.\n";

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
