#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <algorithm>
#include <cmath>
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
     * This example models nitrogen gas flowing through an imported honeycomb triangle mesh.
     * Particles are emitted from a circular source placed below the honeycomb region and
     * then evolve under:
     * - DSMC particle-particle collisions
     * - collisions against the imported mesh collider
     * - sink removal outside the padded simulation domain
     *
     * Emitted particles are sampled from a Maxwell-Boltzmann distribution centered
     * around the configured bulk inflow velocity.
     */
    constexpr T kTemperature           = 300.0f;
    constexpr T kDt                    = 2.5e-6f;
    constexpr std::size_t kBufferSize  = 1000000;
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
    constexpr T kSourceSpacing = 0.1f;

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
     * @brief Imported honeycomb mesh processing and derived domain construction settings.
     *
     * The honeycomb OBJ asset is assumed to already use meter-scale coordinates, so the
     * mesh only needs to be uniformly scaled and re-centered around the world origin.
     *
     * The simulation domain is constructed from the honeycomb mesh axis-aligned bounding
     * box scaled about its center by `kDomainAabbScale`.
     *
     * The particle source is positioned slightly above the lower domain boundary and
     * centered on the honeycomb mesh.
     */
    constexpr T kHoneycombMeshScale = 1.0f;
    constexpr T kDomainAabbScale    = 1.5f;
    constexpr T kSourceInset        = 0.10f;
    constexpr T kSourceRadiusScale  = 0.95f;

    /**
     * @brief Bulk drift velocity assigned to emitted particles.
     *
     * Emission is not drift-free in this example. Newly spawned particles are drawn
     * from a thermal distribution centered around this inflow velocity.
     */
    const Vec3 kBulkVelocity(0.0f, 1000.0f, 0.0f);

    /**
     * @brief Surface interaction settings used by the honeycomb collider.
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
    const atlas::Vector4<T> kParticleColor(0.0f, 0.32f, 1.0f, 0.50f);
    const atlas::Vector4<T> kColliderColor(0.8f, 0.8f, 0.8f, 0.28f);
    const atlas::Vector4<T> kColliderEdgeColor(0.f, 0.f, 0.f, 1.f);
    const atlas::Vector4<T> kViewerBackgroundColor(1.f, 1.f, 1.f, 1.f);

    /**
     * @brief Rendering point size used by Vizkit.
     */
    constexpr T kParticlePointSize = 8.0f;
    constexpr T kGeometryLineWidth = 1.5f;

    constexpr const char* kViewerTitle = "Atlas DSMC Nitrogen Honeycomb Flow";

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
 * @brief Return the filesystem path of the honeycomb OBJ mesh asset.
 *
 * The path is resolved relative to the example source location.
 *
 * @return Filesystem path to the honeycomb mesh OBJ file.
 */
std::filesystem::path
honeycomb_mesh_path() {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()
        / "assets"
        / "honeycomb"
        / "honeycomb.obj";
}

/**
 * @brief Create a rigid world-space unit for the given geometry.
 *
 * The geometry defines the local shape of the object. The sync object stores the
 * rigid transform used by runtime collision queries and Vizkit layers.
 *
 * @param geometry Geometry to wrap in a unit.
 * @param translation World-space translation of the unit origin.
 * @param orientation World-space orientation of the unit.
 * @return Host-shared pointer to the constructed unit.
 */
atlas::UnitHostPtr<T>
make_unit(const atlas::GeometryHostPtr<T>& geometry,
          const Vec3& translation                 = Vec3(0, 0, 0),
          const atlas::Quaternion<T>& orientation = atlas::Quaternion<T>(1, 0, 0, 0)) {
    const auto sync = atlas::Sync<T>::builder()
                          .with_rigid_pose(translation, orientation)
                          .make_host_shared();

    return atlas::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .make_host_shared();
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
 * The source disk is positioned slightly above the lower y boundary of the padded
 * domain and centered below the honeycomb mesh. Its radius is
 * derived from the smaller of the x- and z-extents of the padded domain so it
 * nearly spans the domain cross-section.
 *
 * @param padded_bounds Axis-aligned bounds of the padded simulation domain.
 * @return Host-shared pointer to the circular source geometry.
 */
atlas::GeometryHostPtr<T>
make_source_geometry(const atlas::spatial::AxisAlignedBoundingBox<T>& padded_bounds) {
    const Vec3 extents = padded_bounds.extents();
    const T radius     = std::max(
        T(0.05),
        T(0.5) * std::min(extents.x, extents.z) * config::kSourceRadiusScale);
    const Vec3 center(
        0.0f,
        padded_bounds.lower_corner.y + config::kSourceInset,
        T(0));

    return atlas::geometry::Circle<T>::builder()
        .with_center(center)
        .with_normal(Vec3(0.0f, 1.0f, 0.0f))
        .with_radius(radius)
        .make_host_shared();
}

/**
 * @brief Load, validate, normalize, and build the honeycomb triangle mesh geometry.
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
make_honeycomb_geometry() {
    auto mesh = atlas::geometry::TriangleMesh<T>::builder()
                    .load_from_obj(honeycomb_mesh_path().string())
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
        transformed_triangle.a()                          = (triangle.a() - raw_center) * config::kHoneycombMeshScale;
        transformed_triangle.b()                          = (triangle.b() - raw_center) * config::kHoneycombMeshScale;
        transformed_triangle.c()                          = (triangle.c() - raw_center) * config::kHoneycombMeshScale;

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
        throw std::runtime_error("Failed to build a valid honeycomb triangle mesh.");
    }

    mesh.set_triangles(processed_triangles);
    return atlas::make_host_shared<atlas::geometry::TriangleMesh<T>>(std::move(mesh));
}

/**
 * @brief Build the simulation domain from a scaled mesh AABB.
 *
 * The universe and sink operate on a regular box domain rather than the exact
 * honeycomb mesh volume. This helper expands the mesh bounding box uniformly about
 * its center so the final AABB is `kDomainAabbScale` times the mesh AABB size.
 *
 * @param mesh_bounds Axis-aligned bounds of the honeycomb mesh geometry.
 * @return Host-shared pointer to the scaled box geometry.
 */
atlas::GeometryHostPtr<T>
make_domain_geometry(const atlas::spatial::AxisAlignedBoundingBox<T>& mesh_bounds) {
    const Vec3 center       = mesh_bounds.center();
    const Vec3 half_extents = mesh_bounds.extents() * (config::kDomainAabbScale * T(0.5));

    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(center - half_extents)
        .with_upper_corner(center + half_extents)
        .make_host_shared();
}

/**
 * @brief Construct the wall interaction model used by the honeycomb collider.
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
 * @brief Entry point of the DSMC nitrogen honeycomb-flow example.
 *
 * The program assembles the full runtime pipeline, including:
 * - observer and particle storage
 * - imported honeycomb mesh collider
 * - padded Cartesian simulation domain
 * - spatial hashing searcher
 * - DSMC collision solver and macroscopic measurer
 * - circular source, sink, and static collider unit
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
     * @brief Build the honeycomb mesh, padded domain, and source geometry.
     *
     * The universe is defined over a padded axis-aligned box derived from the
     * honeycomb mesh bounds because the searcher and DSMC solver operate on a regular
     * Cartesian grid.
     */
    const auto honeycomb_geometry = make_honeycomb_geometry();
    const auto honeycomb_bounds   = honeycomb_geometry->bound();
    const auto domain_geometry    = make_domain_geometry(honeycomb_bounds);
    const auto source_geometry    = make_source_geometry(domain_geometry->bound());
    const auto domain_bounds      = domain_geometry->bound();

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
    const auto dsmc_solver = atlas::DsmcSolver<T>::builder()
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
     * The honeycomb collider unit is static. It uses the OBJ orientation directly
     * without tilt or rotational motion.
     */
    const auto domain_unit    = make_unit(domain_geometry);
    const auto source_unit    = make_unit(source_geometry);
    const auto honeycomb_unit = make_unit(honeycomb_geometry);

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
     * exact honeycomb mesh shape.
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
     * @brief Create the collider that reflects particles on the honeycomb triangle mesh.
     *
     * `with_flip(true)` inverts the default sidedness handling so the collision
     * response is applied on the intended side of the imported surface.
     */
    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *honeycomb_unit })
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
     * - collider resolves honeycomb-mesh wall interactions
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
                                          .with_background_color(config::kViewerBackgroundColor)
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
            .with_point_size(config::kParticlePointSize)
            .make_shared());

    /**
     * @brief Render the padded simulation domain as a wireframe reference box.
     */
    // viewer.add_layer(
    //     atlas::vizkit::BoxLayer<T>::builder()
    //         .with_unit(domain_unit)
    //         .make_shared());

    /**
     * @brief Render the imported honeycomb mesh as a semi-transparent reference layer.
     */
    const auto honeycomb_layer = atlas::vizkit::TriangleMeshLayer<T>::builder()
                                     .with_unit(honeycomb_unit)
                                     .make_shared();
    honeycomb_layer->set_color(config::kColliderColor);
    honeycomb_layer->set_edge_color(config::kColliderEdgeColor);
    honeycomb_layer->set_line_width(config::kGeometryLineWidth);
    viewer.add_layer(honeycomb_layer);

    std::cout
        << "Nitrogen DSMC example: 300 K, bulk inflow along +y, circular source below the honeycomb, "
           "flow through a static imported triangle-mesh collider.\n";

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
