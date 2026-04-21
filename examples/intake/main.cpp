#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

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
    constexpr std::size_t kBufferSize  = 500000;
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
    constexpr int kViewerWidth  = 1440;
    constexpr int kViewerHeight = 900;

    // Number of updates to execute when the example is built without Vizkit.
    constexpr int kHeadlessSteps = 1000;

    // Intake mesh import settings.
    //
    // The OBJ export already uses meter-scale coordinates, so keep the mesh at
    // unit scale and only re-center it around the world origin.
    constexpr T kIntakeMeshScale = 1.0f;
    const Vec3 kDomainPadding(.5f, 4.f, 2.f);
    constexpr T kSourceInset         = 0.10f;
    constexpr T kSourceRadiusPad     = 0.18f;
    constexpr T kSourceForwardOffset = 0.45f;

    // Start the intake slightly tilted and keep it rotating slowly so collider
    // motion is visible in both the simulation and Vizkit.
    constexpr T kColliderTiltXRad   = -0.18f;
    constexpr T kColliderTiltZRad   = 0.10f;
    // Vizkit layer updates use the simulation dt, which is very small in this
    // example, so the collider needs a much larger angular speed to look like
    // it is actually rotating on screen.
    constexpr T kColliderSpinRate   = 640.0f;
    const Vec3 kColliderSpinAxis(0.0f, 1.0f, 0.25f);

    // The source emits a thermalized fluid with no prescribed bulk drift.
    const Vec3 kBulkVelocity(0.0f, -5000.0f, 0.0f);

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
    const atlas::Vector4<T> kParticleColor(0.10f, 0.74f, 0.92f, 0.40f);
    const atlas::Vector4<T> kColliderColor(0.92f, 0.96f, 0.98f, 0.8f);
    constexpr const char* kViewerTitle = "Atlas DSMC Nitrogen Intake Flow";

} // namespace config

std::filesystem::path
intake_mesh_path() {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "assets" / "intake"
         / "intake.obj";
}

atlas::UnitHostPtr<T>
make_unit(const atlas::GeometryHostPtr<T>& geometry,
          const Vec3& translation = Vec3(0, 0, 0),
          const atlas::Quaternion<T>& orientation = atlas::Quaternion<T>(1, 0, 0, 0),
          const Vec3* angular_velocity = nullptr) {
    // Units in this example are rigid world-space objects.
    //
    // The geometry defines the local shape, while Sync and the optional angular
    // velocity control pose and runtime motion.
    const auto sync = atlas::Sync<T>::builder()
                          .with_rigid_pose(translation, orientation)
                          .make_host_shared();

    auto builder = atlas::Unit<T>::builder();
    builder.with_geometry(geometry).with_sync(sync);
    if (angular_velocity != nullptr) {
        builder.with_angular_velocity(*angular_velocity);
    }

    return builder.make_host_shared();
}

atlas::Quaternion<T>
make_collider_orientation() {
    return atlas::Quaternion<T>::from_euler_xyz(
        config::kColliderTiltXRad,
        T(0),
        config::kColliderTiltZRad);
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
make_source_geometry(const atlas::spatial::AxisAlignedBoundingBox<T>& bounds) {
    // Emit from a disk slightly below the upper opening and pulled forward
    // toward the intake mouth.
    const Vec3 extents = bounds.extents();
    const T radius = std::max(
        T(0.05),
        T(0.5) * std::min(extents.x, extents.z) - config::kSourceRadiusPad);
    const Vec3 center(0.0f, bounds.upper_corner.y - config::kSourceInset, config::kSourceForwardOffset);

    return atlas::geometry::Circle<T>::builder()
        .with_center(center)
        .with_normal(Vec3(0.0f, -1.0f, 0.0f))
        .with_radius(radius)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_intake_geometry() {
    auto mesh = atlas::geometry::TriangleMesh<T>::builder().load_from_obj(intake_mesh_path().string()).build();

    atlas::spatial::AxisAlignedBoundingBox<T> raw_bounds;
    for (const auto& triangle : mesh.triangles) {
        raw_bounds.merge(triangle.a());
        raw_bounds.merge(triangle.b());
        raw_bounds.merge(triangle.c());
    }

    const Vec3 center = raw_bounds.center();
    atlas::HostBuffer<atlas::TriangleContainer4<T>> triangles;

    // Re-center the imported mesh at the world origin. The OBJ coordinates are
    // already in meters, so only a uniform scale constant is applied.
    for (const auto& triangle : mesh.triangles) {
        atlas::TriangleContainer4<T> transformed = triangle;
        transformed.a()                          = (triangle.a() - center) * config::kIntakeMeshScale;
        transformed.b()                          = (triangle.b() - center) * config::kIntakeMeshScale;
        transformed.c()                          = (triangle.c() - center) * config::kIntakeMeshScale;

        Vec3 normal = atlas::math::cross(
            transformed.b() - transformed.a(),
            transformed.c() - transformed.a());
        const T n2 = normal.length_squared();
        if (!(n2 > T(atlas::eps))) continue;

        transformed.d() = normal * (T(1) / static_cast<T>(std::sqrt(n2)));
        triangles.push_back(transformed);
    }

    if (triangles.empty()) {
        throw std::runtime_error("Failed to build a valid intake triangle mesh.");
    }

    mesh.set_triangles(triangles);
    return atlas::make_host_shared<atlas::geometry::TriangleMesh<T>>(std::move(mesh));
}

atlas::GeometryHostPtr<T>
make_bound_geometry(const atlas::spatial::AxisAlignedBoundingBox<T>& geometry_bounds) {
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(geometry_bounds.lower_corner - config::kDomainPadding)
        .with_upper_corner(geometry_bounds.upper_corner + config::kDomainPadding)
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
    const auto intake_geometry = make_intake_geometry();
    const auto intake_bounds   = intake_geometry->bound();
    const auto domain_geometry = make_bound_geometry(intake_bounds);
    const auto source_geometry = make_source_geometry(domain_geometry->bound());
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
    const auto collider_orientation = make_collider_orientation();
    const auto collider_spin        = config::kColliderSpinAxis * config::kColliderSpinRate;

    const auto domain_unit = make_unit(domain_geometry);
    const auto source_unit = make_unit(source_geometry);
    const auto intake_unit = make_unit(
        intake_geometry,
        Vec3(0, 0, 0),
        collider_orientation,
        &collider_spin);

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

    // Sink removes particles that leave the intake's padded bounding box.
    const auto sink = atlas::fluid::Sink<T>::builder()
                          .with_units(atlas::HostBuffer<atlas::Unit<T>> { *domain_unit })
                          .with_fluid(fluid)
                          .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {
                              atlas::fluid::DespawnType::Volume,
                          })
                          .with_despawn_operator(atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Volume))
                          .with_flip(true)
                          .make_host_shared();

    // Collider applies diffuse wall reflection on the imported intake mesh.
    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *intake_unit })
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
    // 3. collider resolves intake wall hits
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

    // Render the imported intake mesh as a semi-transparent wireframe reference.
    const auto intake_layer = atlas::vizkit::TriangleMeshLayer<T>::builder()
                                  .with_unit(intake_unit)
                                  .make_shared();
    intake_layer->set_color(config::kColliderColor);
    viewer.add_layer(intake_layer);

    std::cout
        << "Nitrogen DSMC example: 300 K, zero bulk drift, circular source above the intake, flow around an imported triangle-mesh collider.\n";

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
