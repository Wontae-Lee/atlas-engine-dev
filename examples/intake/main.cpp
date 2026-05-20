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

    /**
     * @brief Thermodynamic, molecular, and time-integration parameters used by the example.
     *
     * This example models nitrogen gas flowing around an imported intake triangle mesh.
     * Particles are emitted from a circular source placed above the intake region and
     * then evolve under:
     * - DSMC particle-particle collisions
     * - collisions against the imported mesh collider
     * - sink removal outside the padded simulation domain
     *
     * Emitted particles are sampled from a Maxwell-Boltzmann distribution centered
     * around the configured bulk inflow velocity.
     */
    constexpr T kTemperature           = 300.0f;
    constexpr T kDt                    = 2.5e-5f;
    constexpr std::size_t kBufferSize  = 500000;
    constexpr T kNitrogenMolecularMass = 4.651734e-26f;
    constexpr T kNitrogenDiameter      = 4.17e-10f;

    /**
     * @brief Spatial discretization and source sampling density parameters.
     *
     * `kCellSize` defines the Cartesian background grid resolution shared by the
     * spatial hashing searcher and the DSMC collision solver.
     *
     * `kSourceSpacing` controls how densely the circular source surface is sampled
     * when generating candidate emission positions. Smaller values produce a denser
     * particle stream but also increase runtime cost.
     */
    constexpr T kCellSize      = 0.25f;
    constexpr T kSourceSpacing = 0.15f;

    /**
     * @brief Viewer size used by the Vizkit execution path.
     */
    constexpr int kViewerWidth  = 1440;
    constexpr int kViewerHeight = 900;

    /**
     * @brief Headless execution and observer preallocation settings.
     *
     * `kHeadlessSteps` specifies how many simulation steps are executed when Vizkit
     * is not enabled.
     *
     * `kObserverReserveCount` pre-reserves observer metric storage to reduce dynamic
     * allocations during the run.
     */
    constexpr int kHeadlessSteps                = 1000;
    constexpr std::size_t kObserverReserveCount = 4096;

    /**
     * @brief Imported intake mesh processing and derived domain construction settings.
     *
     * The intake OBJ asset is assumed to already use meter-scale coordinates, so the
     * mesh only needs to be uniformly scaled and re-centered around the world origin.
     *
     * The simulation domain is constructed from the intake mesh axis-aligned bounding
     * box expanded by `kDomainPadding`.
     *
     * The particle source is positioned slightly below the upper domain boundary and
     * offset forward toward the intake entrance.
     */
    constexpr T kIntakeMeshScale = 1.0f;
    const Vec3 kDomainPadding(0.5f, 4.0f, 2.0f);
    constexpr T kSourceInset         = 0.10f;
    constexpr T kSourceRadiusPad     = 0.18f;
    constexpr T kSourceForwardOffset = 0.45f;

    /**
     * @brief Initial collider orientation and rotational motion settings.
     *
     * The intake mesh is initialized with a small tilt and continuously rotated so
     * that collider motion remains visible in both the simulation and Vizkit.
     *
     * Because the simulation time step is very small, the visual angular speed must
     * be comparatively large to make the motion perceptible.
     */
    constexpr T kColliderTiltXRad = -0.18f;
    constexpr T kColliderTiltZRad = 0.10f;
    constexpr T kColliderSpinRate = 640.0f;
    const Vec3 kColliderSpinAxis(0.0f, 1.0f, 0.25f);

    /**
     * @brief Bulk drift velocity assigned to emitted particles.
     *
     * Emission is not drift-free in this example. Newly spawned particles are drawn
     * from a thermal distribution centered around this inflow velocity.
     */
    const Vec3 kBulkVelocity(0.0f, -5000.0f, 0.0f);

    /**
     * @brief Surface interaction settings used by the intake collider.
     *
     * The collider uses a fully diffuse reflection model:
     * - cosine-weighted hemisphere sampling for outgoing directions
     * - TMAC = 1, which always selects the diffuse branch
     * - restitution = 1, which preserves incident speed magnitude
     */
    constexpr atlas::system::DiffuseSampling kDiffuseSampling
        = atlas::system::DiffuseSampling::CosineWeighted;
    constexpr T kRestitution                     = 1.0f;
    constexpr T kTangentialMomentumAccommodation = 1.0f;

    /**
     * @brief DSMC kernel type used for particle-particle collisions.
     */
    constexpr atlas::system::DsmcKernelType kDsmcKernelType
        = atlas::system::DsmcKernelType::hard_sphere;

    /**
     * @brief Measurement mode used for macroscopic diagnostics.
     *
     * `Field` stores measured quantities on the universe grid. This is sufficient
     * for this example because the result is used only as a cell-wise diagnostic and
     * is not written back into per-particle state.
     */
    constexpr atlas::MeasureModeType kMeasureMode = atlas::MeasureModeType::Field;

    /**
     * @brief Rendering colors and window title used by Vizkit.
     */
    const atlas::Vector4<T> kParticleColor(0.10f, 0.74f, 0.92f, 0.40f);
    const atlas::Vector4<T> kColliderColor(0.92f, 0.96f, 0.98f, 0.80f);
    constexpr const char* kViewerTitle = "Atlas DSMC Nitrogen Intake Flow";

} // namespace config

/**
 * @brief Return the output directory used for observer CSV export.
 *
 * The path is resolved relative to the current source file so exported observer
 * diagnostics are written next to the example source tree.
 *
 * @return Filesystem path to the observer output directory.
 */
std::filesystem::path
observer_output_path() {
    return std::filesystem::path(__FILE__).parent_path() / "observer_output";
}

/**
 * @brief Return the filesystem path of the intake OBJ mesh asset.
 *
 * The path is resolved relative to the example source location.
 *
 * @return Filesystem path to the intake mesh OBJ file.
 */
std::filesystem::path
intake_mesh_path() {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()
        / "assets"
        / "intake"
        / "intake.obj";
}

/**
 * @brief Create a rigid world-space unit for the given geometry.
 *
 * The geometry defines the local shape of the object. The sync object stores the
 * rigid transform used by runtime collision queries and Vizkit layers. When an
 * angular velocity is provided, the unit also carries rotational motion.
 *
 * @param geometry Geometry to wrap in a unit.
 * @param translation World-space translation of the unit origin.
 * @param orientation World-space orientation of the unit.
 * @param angular_velocity Optional angular velocity vector. When null, the unit
 *        is created without rotational velocity.
 * @return Host-shared pointer to the constructed unit.
 */
atlas::UnitHostPtr<T>
make_unit(const atlas::GeometryHostPtr<T>& geometry,
          const Vec3& translation                 = Vec3(0, 0, 0),
          const atlas::Quaternion<T>& orientation = atlas::Quaternion<T>(1, 0, 0, 0),
          const Vec3* angular_velocity            = nullptr) {
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

/**
 * @brief Build the initial orientation applied to the intake collider.
 *
 * The intake starts with a fixed tilt around the x- and z-axes.
 *
 * @return Quaternion representing the initial collider orientation.
 */
atlas::Quaternion<T>
make_collider_orientation() {
    return atlas::Quaternion<T>::from_euler_xyz(
        config::kColliderTiltXRad,
        T(0),
        config::kColliderTiltZRad);
}

/**
 * @brief Create the fluid container and configure its species and emission generator.
 *
 * Atlas fluids store:
 * - material and species properties
 * - host-configured generator models
 * - fixed-capacity particle state buffers
 *
 * This example uses a single nitrogen species, so both the material property
 * table and the generator table contain exactly one entry.
 *
 * @param observer Observer used to collect runtime metrics.
 * @return Host-shared pointer to the configured fluid.
 */
atlas::FluidHostPtr<T>
make_fluid(const atlas::ObserverHostPtr& observer) {
    atlas::HostBuffer<atlas::MaterialProperties<T>> material_properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    /**
     * @brief Configure the nitrogen material record required by the DSMC solver.
     *
     * The hard-sphere kernel reads at least:
     * - molecular_mass
     * - reference_diameter
     *
     * `species_id` is also assigned so the runtime retains an explicit species
     * label for the particle population.
     */
    material_properties[0] = atlas::MaterialProperties<T>::builder()
                                 .with_type(atlas::MaterialType::Molecule)
                                 .with_mass(config::kNitrogenMolecularMass)
                                 .with_molecular_mass(config::kNitrogenMolecularMass)
                                 .with_species_id(0)
                                 .with_reference_diameter(config::kNitrogenDiameter)
                                 .build();

    /**
     * @brief Configure the Maxwell-Boltzmann emission generator with imposed drift.
     *
     * Newly emitted particles are sampled from a thermal distribution centered on
     * the configured bulk inflow velocity.
     */
    generators[0] = atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
                        .with_temperature(config::kTemperature)
                        .with_molecular_mass(config::kNitrogenMolecularMass)
                        .with_bulk_velocity(config::kBulkVelocity)
                        .with_seed(42u)
                        .make_host_shared();

    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(config::kBufferSize)
        .with_properties(material_properties)
        .with_generators(generators)
        .with_observer(observer)
        .make_host_shared();
}

/**
 * @brief Build the circular source geometry from the padded simulation bounds.
 *
 * The source disk is positioned slightly below the upper y boundary of the padded
 * domain and shifted forward along z toward the intake opening. Its radius is
 * derived from the smaller of the x- and z-extents of the padded domain with an
 * additional padding margin removed.
 *
 * @param padded_bounds Axis-aligned bounds of the padded simulation domain.
 * @return Host-shared pointer to the circular source geometry.
 */
atlas::GeometryHostPtr<T>
make_source_geometry(const atlas::spatial::AxisAlignedBoundingBox<T>& padded_bounds) {
    const Vec3 extents = padded_bounds.extents();
    const T radius     = std::max(
        T(0.05),
        T(0.5) * std::min(extents.x, extents.z) - config::kSourceRadiusPad);
    const Vec3 center(
        0.0f,
        padded_bounds.upper_corner.y - config::kSourceInset,
        config::kSourceForwardOffset);

    return atlas::geometry::Circle<T>::builder()
        .with_center(center)
        .with_normal(Vec3(0.0f, -1.0f, 0.0f))
        .with_radius(radius)
        .make_host_shared();
}

/**
 * @brief Load, validate, normalize, and build the intake triangle mesh geometry.
 *
 * The OBJ mesh is loaded from disk, its raw bounds are computed, and the mesh is
 * re-centered around the world origin. The imported coordinates are assumed to
 * already be in meters, so only a uniform scale factor is applied.
 *
 * Degenerate triangles are discarded. For each retained triangle, a normalized
 * geometric normal is recomputed and stored in the fourth slot of the triangle
 * container.
 *
 * @throws std::runtime_error Thrown when no valid triangles remain after
 *         filtering and normal reconstruction.
 * @return Host-shared pointer to the processed triangle mesh geometry.
 */
atlas::GeometryHostPtr<T>
make_intake_geometry() {
    auto mesh = atlas::geometry::TriangleMesh<T>::builder()
                    .load_from_obj(intake_mesh_path().string())
                    .build();

    atlas::spatial::AxisAlignedBoundingBox<T> raw_bounds;
    for (const auto& triangle : mesh.triangles) {
        raw_bounds.merge(triangle.a());
        raw_bounds.merge(triangle.b());
        raw_bounds.merge(triangle.c());
    }

    const Vec3 raw_center = raw_bounds.center();
    atlas::HostBuffer<atlas::TriangleContainer4<T>> processed_triangles;

    for (const auto& triangle : mesh.triangles) {
        atlas::TriangleContainer4<T> transformed_triangle = triangle;
        transformed_triangle.a()                          = (triangle.a() - raw_center) * config::kIntakeMeshScale;
        transformed_triangle.b()                          = (triangle.b() - raw_center) * config::kIntakeMeshScale;
        transformed_triangle.c()                          = (triangle.c() - raw_center) * config::kIntakeMeshScale;

        Vec3 geometric_normal = atlas::math::cross(
            transformed_triangle.b() - transformed_triangle.a(),
            transformed_triangle.c() - transformed_triangle.a());

        const T normal_length_squared = geometric_normal.length_squared();
        if (!(normal_length_squared > T(atlas::eps))) {
            continue;
        }

        transformed_triangle.d()
            = geometric_normal * (T(1) / static_cast<T>(std::sqrt(normal_length_squared)));
        processed_triangles.push_back(transformed_triangle);
    }

    if (processed_triangles.empty()) {
        throw std::runtime_error("Failed to build a valid intake triangle mesh.");
    }

    mesh.set_triangles(processed_triangles);
    return atlas::make_host_shared<atlas::geometry::TriangleMesh<T>>(std::move(mesh));
}

/**
 * @brief Build the padded axis-aligned simulation domain from mesh bounds.
 *
 * The universe and sink operate on a regular box domain rather than the exact
 * intake mesh volume. This helper expands the mesh bounding box by the configured
 * padding in each axis.
 *
 * @param mesh_bounds Axis-aligned bounds of the intake mesh geometry.
 * @return Host-shared pointer to the padded box geometry.
 */
atlas::GeometryHostPtr<T>
make_domain_geometry(const atlas::spatial::AxisAlignedBoundingBox<T>& mesh_bounds) {
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(mesh_bounds.lower_corner - config::kDomainPadding)
        .with_upper_corner(mesh_bounds.upper_corner + config::kDomainPadding)
        .make_host_shared();
}

/**
 * @brief Construct the wall interaction model used by the intake collider.
 *
 * Diffuse reflection randomizes the outgoing direction according to the selected
 * sampling model. The reflected speed magnitude remains controlled by the
 * restitution parameter.
 *
 * @return Configured collider surface interaction descriptor.
 */
atlas::system::ColliderSurfaceInteraction<T>
make_collider_interaction() {
    return atlas::system::ColliderSurfaceInteraction<T>::builder()
        .with_diffuse_sampling(config::kDiffuseSampling)
        .with_restitution(config::kRestitution)
        .with_tangential_momentum_accommodation(config::kTangentialMomentumAccommodation)
        .with_temperature(config::kTemperature)
        .build();
}

} // namespace

/**
 * @brief Entry point of the DSMC nitrogen intake-flow example.
 *
 * The program assembles the full runtime pipeline, including:
 * - observer and particle storage
 * - imported intake mesh collider
 * - padded Cartesian simulation domain
 * - spatial hashing searcher
 * - DSMC collision solver and macroscopic measurer
 * - circular source, sink, and rotating collider unit
 * - optional Vizkit visualization
 *
 * When Vizkit is enabled, the example runs interactively. Otherwise the same
 * simulation pipeline is executed headlessly for a fixed number of steps.
 *
 * @return Process exit code.
 */
int
main() {
    const auto observer = atlas::Observer::builder()
                              .with_source_sensor_matrics(config::kObserverReserveCount)
                              .with_sink_sensor_matrics(config::kObserverReserveCount)
                              .make_host_shared();

    /**
     * @brief Create the fluid first because most runtime subsystems depend on it.
     */
    const auto fluid = make_fluid(observer);

    /**
     * @brief Build the intake mesh, padded domain, and source geometry.
     *
     * The universe is defined over a padded axis-aligned box derived from the
     * intake mesh bounds because the searcher and DSMC solver operate on a regular
     * Cartesian grid.
     */
    const auto intake_geometry = make_intake_geometry();
    const auto intake_bounds   = intake_geometry->bound();
    const auto domain_geometry = make_domain_geometry(intake_bounds);
    const auto source_geometry = make_source_geometry(domain_geometry->bound());
    const auto domain_bounds   = domain_geometry->bound();

    /**
     * @brief Create the universe that owns the regular background cell grid.
     *
     * The spatial hashing searcher and DSMC solver both reuse this grid.
     */
    const auto universe = atlas::Universe<T>::builder()
                              .with_lower_corner(domain_bounds.lower_corner)
                              .with_upper_corner(domain_bounds.upper_corner)
                              .with_cell_size(config::kCellSize)
                              .with_observer(observer)
                              .make_host_shared();

    /**
     * @brief Create the spatial hashing searcher used for particle binning.
     *
     * Each update maps active particles into grid cells so the DSMC solver can
     * efficiently query local collision neighborhoods.
     */
    const auto searcher = atlas::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    /**
     * @brief Build the DSMC solver responsible for particle-particle collisions.
     */
    const auto dsmc_solver = atlas::DsmcCellSequentialSolver<T>::builder()
                                 .with_universe(universe)
                                 .with_fluid(fluid)
                                 .with_searcher(searcher)
                                 .with_kernel_type(config::kDsmcKernelType)
                                 .make_host_shared();

    /**
     * @brief Build the measurer that computes macroscopic cell fields.
     *
     * After particles have been assigned to cells, the measurer derives cell-wise
     * diagnostic quantities such as bulk velocity and temperature.
     */
    const auto measurer = atlas::BoltzmanMeasurer<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_measure_mode(config::kMeasureMode)
                              .make_host_shared();

    /**
     * @brief Build the orchestrator that owns the internal search-measure-solve sequence.
     */
    const auto orchestrator = atlas::Orchestrator<T>::builder()
                                  .with_universe(universe)
                                  .with_fluid(fluid)
                                  .with_searcher(searcher)
                                  .with_measurer(measurer)
                                  .with_solver(dsmc_solver)
                                  .make_host_shared();

    /**
     * @brief Build the shared units used by runtime systems and Vizkit.
     *
     * The intake collider unit is initialized with a tilted orientation and a
     * constant angular velocity so the mesh rotates during the simulation.
     */
    const auto collider_orientation = make_collider_orientation();
    const auto collider_spin        = config::kColliderSpinAxis * config::kColliderSpinRate;

    const auto domain_unit = make_unit(domain_geometry);
    const auto source_unit = make_unit(source_geometry);
    const auto intake_unit = make_unit(
        intake_geometry,
        Vec3(0, 0, 0),
        collider_orientation,
        &collider_spin);

    /**
     * @brief Create the surface source that emits particles from the circular disk.
     *
     * The source samples positions on the disk surface and assigns velocities using
     * the configured thermal generator with imposed bulk inflow.
     */
    const auto source = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { *source_unit })
                            .with_fluid(fluid)
                            .with_observer(observer)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                                atlas::fluid::SpawnType::Surface,
                            })
                            .with_spawn_operator(
                                atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Surface))
                            .with_spacing(config::kSourceSpacing)
                            .with_temperature(config::kTemperature)
                            .make_host_shared();

    /**
     * @brief Create the sink that removes particles leaving the padded domain box.
     *
     * The sink operates on the padded axis-aligned bounding box rather than the
     * exact intake mesh shape.
     */
    const auto sink = atlas::fluid::Sink<T>::builder()
                          .with_units(atlas::HostBuffer<atlas::Unit<T>> { *domain_unit })
                          .with_fluid(fluid)
                          .with_observer(observer)
                          .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {
                              atlas::fluid::DespawnType::Volume,
                          })
                          .with_despawn_operator(
                              atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Volume))
                          .with_flip(true)
                          .make_host_shared();

    /**
     * @brief Create the collider that reflects particles on the intake triangle mesh.
     *
     * `with_flip(true)` inverts the default sidedness handling so the collision
     * response is applied on the intended side of the imported surface.
     */
    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *intake_unit })
                              .with_surface_interactions(
                                  atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> {
                                      make_collider_interaction(),
                                  })
                              .with_flip(true)
                              .make_host_shared();

    /**
     * @brief Assemble the top-level simulation system.
     *
     * The per-update execution order is:
     * - source emits new particles
     * - orchestrator performs search, measurement, and DSMC collision solving
     * - collider resolves intake-mesh wall interactions
     * - sink removes particles that escaped the padded simulation domain
     */
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
    /**
     * @brief Run the interactive Vizkit visualization path.
     */
    atlas::vizkit::Viewer<T> viewer = atlas::vizkit::Viewer<T>::builder()
                                          .with_system(system)
                                          .with_title(config::kViewerTitle)
                                          .with_size(config::kViewerWidth, config::kViewerHeight)
                                          .build();

    /**
     * @brief Frame the initial camera around the padded simulation bounds.
     */
    viewer.camera().fit_bounds(domain_bounds.lower_corner, domain_bounds.upper_corner);

    /**
     * @brief Render the live particle cloud.
     */
    viewer.add_layer(
        atlas::vizkit::ParticleLayer<T>::builder()
            .with_system(system)
            .with_color(config::kParticleColor)
            .make_shared());

    /**
     * @brief Render the padded simulation domain as a wireframe reference box.
     */
    viewer.add_layer(
        atlas::vizkit::BoxLayer<T>::builder()
            .with_unit(domain_unit)
            .make_shared());

    /**
     * @brief Render the imported intake mesh as a semi-transparent reference layer.
     */
    const auto intake_layer = atlas::vizkit::TriangleMeshLayer<T>::builder()
                                  .with_unit(intake_unit)
                                  .make_shared();
    intake_layer->set_color(config::kColliderColor);
    viewer.add_layer(intake_layer);

    std::cout
        << "Nitrogen DSMC example: 300 K, bulk inflow along -y, circular source above the intake, "
           "flow around a rotating imported triangle-mesh collider.\n";

    const int exit_code = viewer.run();
    observer->export_csv(observer_output_path());
    return exit_code;
#else
    /**
     * @brief Run the headless fallback path for non-Vizkit builds.
     *
     * This path keeps the example runnable in environments without OpenGL or GLFW
     * while still exercising the same simulation pipeline.
     */
    for (int step = 0; step < config::kHeadlessSteps; ++step) {
        system->update();
    }

    std::cout
        << "Nitrogen DSMC example ran headlessly for " << config::kHeadlessSteps << " steps.\n"
        << "Active particles: " << fluid->particle_count() << '\n';

    observer->export_csv(observer_output_path());
    return 0;
#endif
}
