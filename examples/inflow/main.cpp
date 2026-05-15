#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <filesystem>
#include <iostream>

namespace {

using T    = float;
using Vec3 = atlas::Vector3<T>;

namespace config {

    /**
     * @brief Gas model and time-integration parameters used by the entire example.
     *
     * This scene models nitrogen inside a finite cylindrical region. Particles are
     * emitted from a circular source near the +z side of the cylinder with no
     * prescribed bulk drift, then evolve under:
     * - DSMC particle-particle collisions
     * - diffuse reflections against the inner cylinder wall
     * - sink removal once they leave the cylinder's axis-aligned bounding box
     */
    constexpr T kTemperature           = 300.0f;
    constexpr T kDt                    = 2.5e-5f;
    constexpr std::size_t kBufferSize  = 200000;
    constexpr T kNitrogenMolecularMass = 4.651734e-26f;
    constexpr T kNitrogenDiameter      = 4.17e-10f;

    /**
     * @brief Spatial discretization and source emission density settings.
     *
     * `kCellSize` defines the Cartesian background grid resolution used by both
     * the spatial hashing searcher and the DSMC collision solver.
     *
     * `kSourceSpacing` controls how densely the circular source surface is sampled
     * for candidate emission positions. A smaller spacing emits more particles per
     * update, which increases visible density and runtime cost at the same time.
     */
    constexpr T kCellSize      = 0.25f;
    constexpr T kSourceSpacing = 0.15f;

    /**
     * @brief Viewer presentation settings used only by the Vizkit path.
     */
    constexpr int kViewerWidth    = 1440;
    constexpr int kViewerHeight   = 900;
    constexpr int kCylinderSlices = 72;

    /**
     * @brief Headless execution and observer preallocation settings.
     *
     * `kHeadlessSteps` is the number of simulation updates executed when the
     * example is built without Vizkit support.
     *
     * `kObserverReserveCount` pre-reserves observer-side metric storage to reduce
     * dynamic reallocations during the run.
     */
    constexpr int kHeadlessSteps                = 1000;
    constexpr std::size_t kObserverReserveCount = 4096;

    /**
     * @brief Finite cylinder geometry that defines the main flow region.
     *
     * The cylinder is centered at the world origin and aligned with the z-axis.
     * This example uses an open cylinder, so the curved side wall is present but
     * the end caps are not part of the collider geometry.
     */
    const Vec3 kCylinderCenter(0, 0, 0);
    constexpr T kCylinderRadius = 1.0f;
    constexpr T kCylinderHeight = 8.0f;

    /**
     * @brief Circular source disk placed near the +z side of the cylinder.
     *
     * The source emits particles from a surface rather than from a finite volume.
     * Its normal points along +z, so the source plane is parallel to the xy-plane.
     */
    const Vec3 kSourceCenter(0.0f, 0.0f, 3.6f);
    const Vec3 kSourceNormal(0.0f, 0.0f, 1.0f);
    constexpr T kSourceRadius = 1.f;

    /**
     * @brief Prescribed bulk drift velocity of emitted particles.
     *
     * The source uses zero bulk drift in this example, so the injected velocity
     * distribution is purely thermal.
     */
    const Vec3 kBulkVelocity(0.0f, 0.0f, 0.0f);

    /**
     * @brief Surface interaction parameters used by the cylinder collider.
     *
     * The wall model is fully diffuse:
     * - cosine-weighted hemisphere sampling is used for outgoing directions
     * - TMAC = 1 selects the fully diffuse branch
     * - restitution = 1 preserves the incident speed magnitude
     */
    constexpr atlas::system::DiffuseSampling kDiffuseSampling
        = atlas::system::DiffuseSampling::CosineWeighted;
    constexpr T kRestitution                     = 1.0f;
    constexpr T kTangentialMomentumAccommodation = 1.0f;

    /**
     * @brief DSMC kernel used for particle-particle collisions.
     */
    constexpr atlas::system::DsmcKernelType kDsmcKernelType
        = atlas::system::DsmcKernelType::hard_sphere;

    /**
     * @brief Measurement mode used for macroscopic diagnostics.
     *
     * `Field` stores measured quantities on the universe grid. That is sufficient
     * for this example because the result is used only as a cell-wise diagnostic
     * and is not written back into per-particle thermodynamic state.
     */
    constexpr atlas::MeasureModeType kMeasureMode = atlas::MeasureModeType::Field;

    /**
     * @brief Basic visual styling for the particle cloud and cylinder rendering.
     */
    const atlas::Vector4<T> kParticleColor(0.10f, 0.74f, 0.92f, 0.80f);
    const atlas::Vector4<T> kCylinderColor(0.92f, 0.96f, 0.98f, 0.28f);
    constexpr const char* kViewerTitle = "Atlas DSMC Nitrogen In-Cylinder Flow";

} // namespace config

/**
 * @brief Return the output directory used for observer CSV export.
 *
 * The export path is resolved relative to the current source file so diagnostic
 * output is written next to the example source tree.
 *
 * @return Filesystem path of the observer output directory.
 */
std::filesystem::path
observer_output_path() {
    return std::filesystem::path(__FILE__).parent_path() / "observer_output";
}

/**
 * @brief Wrap a geometry object in a static world-space unit.
 *
 * In this example, all units are fixed rigid objects. The geometry supplies the
 * analytic shape, while the sync object stores the rigid transform consumed by
 * collider queries and Vizkit geometry layers.
 *
 * @param geometry Geometry to wrap.
 * @return Host-shared pointer to the constructed unit.
 */
atlas::UnitHostPtr<T>
make_unit(const atlas::GeometryHostPtr<T>& geometry) {
    const auto sync = atlas::Sync<T>::builder()
                          .with_rigid_pose(Vec3(0, 0, 0), atlas::Quaternion<T>(1, 0, 0, 0))
                          .make_host_shared();

    return atlas::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .make_host_shared();
}

/**
 * @brief Create the fluid object and configure its species and generators.
 *
 * Atlas fluids own:
 * - species and material properties
 * - host-configured generator models
 * - fixed-capacity particle state buffers
 *
 * This example contains exactly one species, nitrogen, so both the material
 * property table and the generator table contain a single entry.
 *
 * @param observer Observer used for runtime metric collection.
 * @return Host-shared pointer to the configured fluid.
 */
atlas::FluidHostPtr<T>
make_fluid(const atlas::ObserverHostPtr& observer) {
    atlas::HostBuffer<atlas::MaterialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    /**
     * @brief Configure the nitrogen material record used by the DSMC solver.
     *
     * The hard-sphere collision kernel reads at least:
     * - molecular_mass
     * - reference_diameter
     *
     * `species_id` is also installed so the runtime retains an explicit species
     * identifier for the particle population.
     */
    properties[0] = atlas::MaterialProperties<T>::builder()
                        .with_type(atlas::MaterialType::Molecule)
                        .with_mass(config::kNitrogenMolecularMass)
                        .with_molecular_mass(config::kNitrogenMolecularMass)
                        .with_species_id(0)
                        .with_reference_diameter(config::kNitrogenDiameter)
                        .build();

    /**
     * @brief Configure the thermal emission velocity generator.
     *
     * A Maxwell-Boltzmann generator is used with zero bulk drift, so newly emitted
     * particles are sampled from a purely thermal velocity distribution.
     */
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
        .with_observer(observer)
        .make_host_shared();
}

/**
 * @brief Build the circular source geometry.
 *
 * The source is a disk near the +z side of the cylinder and emits particles from
 * its surface into the simulation.
 *
 * @return Host-shared pointer to the circular source geometry.
 */
atlas::GeometryHostPtr<T>
make_source_geometry() {
    return atlas::geometry::Circle<T>::builder()
        .with_center(config::kSourceCenter)
        .with_normal(config::kSourceNormal)
        .with_radius(config::kSourceRadius)
        .make_host_shared();
}

/**
 * @brief Build the main cylinder geometry used by the collider.
 *
 * The cylinder is centered at the origin, aligned with the z-axis, and marked as
 * open so only the curved wall surface participates in collision handling.
 *
 * @return Host-shared pointer to the cylinder geometry.
 */
atlas::GeometryHostPtr<T>
make_cylinder_geometry() {
    return atlas::geometry::Cylinder<T>::builder()
        .with_center(config::kCylinderCenter)
        .with_radius(config::kCylinderRadius)
        .with_height(config::kCylinderHeight)
        .with_open(true)
        .make_host_shared();
}

/**
 * @brief Build an axis-aligned bounding box for the given geometry.
 *
 * This helper is used to derive a simple Cartesian domain from the cylinder
 * geometry. The resulting box is used by the universe and by the sink that
 * removes particles once they leave the bounded simulation region.
 *
 * @param geometry Geometry whose bounds are queried.
 * @return Host-shared pointer to a box representing the geometry bounds.
 */
atlas::GeometryHostPtr<T>
make_bound_geometry(const atlas::GeometryHostPtr<T>& geometry) {
    const auto bounds = geometry->bound();
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(bounds.lower_corner)
        .with_upper_corner(bounds.upper_corner)
        .make_host_shared();
}

/**
 * @brief Build the wall interaction model used by the cylinder collider.
 *
 * Diffuse reflection randomizes the outgoing direction according to the selected
 * sampling model. The outgoing speed magnitude remains controlled by the
 * restitution parameter inside `ColliderSurfaceInteraction`.
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
 * @brief Entry point of the in-cylinder DSMC nitrogen example.
 *
 * The program builds the full simulation pipeline, including:
 * - observer and particle storage
 * - universe bounds and spatial searcher
 * - DSMC collision solver and macroscopic measurer
 * - circular source, sink, and inner-wall cylinder collider
 * - optional Vizkit visualization
 *
 * When Vizkit is enabled, the simulation runs interactively. Otherwise the same
 * runtime pipeline is executed headlessly for a fixed number of updates.
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
     * @brief Build the main cylinder, its axis-aligned bounding box, and the source.
     *
     * The universe is defined over the cylinder bounding box rather than the exact
     * curved cylinder volume because the searcher and solver operate on a regular
     * Cartesian grid.
     */
    const auto cylinder_geometry = make_cylinder_geometry();
    const auto domain_geometry   = make_bound_geometry(cylinder_geometry);
    const auto source_geometry   = make_source_geometry();
    const auto bounds            = domain_geometry->bound();

    /**
     * @brief Create the universe that owns the regular background cell grid.
     *
     * The spatial hashing searcher and DSMC solver both reuse this same grid.
     */
    const auto universe = atlas::Universe<T>::builder()
                              .with_lower_corner(bounds.lower_corner)
                              .with_upper_corner(bounds.upper_corner)
                              .with_cell_size(config::kCellSize)
                              .with_observer(observer)
                              .make_host_shared();

    /**
     * @brief Create the spatial hashing searcher used for neighborhood lookup.
     *
     * Each update maps active particles into grid cells so the DSMC solver can
     * find local collision neighborhoods efficiently.
     */
    const auto searcher = atlas::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    /**
     * @brief Build the DSMC solver that handles particle-particle collisions.
     */
    const auto dsmc_solver = atlas::DsmcNtcSolver<T>::builder()
                                 .with_universe(universe)
                                 .with_fluid(fluid)
                                 .with_searcher(searcher)
                                 .with_kernel_type(config::kDsmcKernelType)
                                 .make_host_shared();

    /**
     * @brief Build the measurer that computes macroscopic cell fields.
     *
     * After particles are assigned to cells, the measurer derives diagnostic field
     * quantities such as bulk velocity and temperature.
     */
    const auto measurer = atlas::BoltzmanMeasurer<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_measure_mode(config::kMeasureMode)
                              .make_host_shared();

    /**
     * @brief Build the orchestrator that owns the internal runtime sequence.
     *
     * The orchestrator performs:
     * - search
     * - measure
     * - solve
     */
    const auto orchestrator = atlas::Orchestrator<T>::builder()
                                  .with_universe(universe)
                                  .with_fluid(fluid)
                                  .with_searcher(searcher)
                                  .with_measurer(measurer)
                                  .with_solver(dsmc_solver)
                                  .make_host_shared();

    /**
     * @brief Wrap each geometry in a unit so runtime systems and Vizkit can share it.
     */
    const auto domain_unit   = make_unit(domain_geometry);
    const auto source_unit   = make_unit(source_geometry);
    const auto cylinder_unit = make_unit(cylinder_geometry);

    /**
     * @brief Create the surface source that emits thermal particles from the disk.
     *
     * The source samples positions on the circular surface and assigns thermal
     * velocities according to the configured generator.
     */
    const auto source = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { *source_unit })
                            .with_fluid(fluid)
                            .with_observer(observer)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                                atlas::fluid::SpawnType::Surface,
                            })
                            .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Surface))
                            .with_spacing(config::kSourceSpacing)
                            .with_temperature(config::kTemperature)
                            .make_host_shared();

    /**
     * @brief Create the sink that removes particles leaving the bounding box.
     *
     * The sink does not test against the exact cylinder volume. Instead it uses
     * the cylinder's axis-aligned bounding box as the valid simulation region and
     * removes particles once they move outside that box.
     */
    const auto sink = atlas::fluid::Sink<T>::builder()
                          .with_units(atlas::HostBuffer<atlas::Unit<T>> { *domain_unit })
                          .with_fluid(fluid)
                          .with_observer(observer)
                          .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {
                              atlas::fluid::DespawnType::Volume,
                          })
                          .with_despawn_operator(atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Volume))
                          .with_flip(true)
                          .make_host_shared();

    /**
     * @brief Create the collider that reflects particles against the inner cylinder wall.
     *
     * `with_flip(true)` makes the collider interpret the cylinder as an inward-facing
     * boundary so the response is applied for particles moving inside the cylindrical
     * flow region rather than outside it.
     */
    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *cylinder_unit })
                              .with_surface_interactions(
                                  atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> {
                                      make_collider_interaction(),
                                  })
                              .with_flip(true)
                              .make_host_shared();

    /**
     * @brief Assemble the top-level runtime system.
     *
     * The per-update execution order is:
     * - source emits new particles
     * - orchestrator performs search, measurement, and DSMC collision solving
     * - collider resolves inner-wall cylinder interactions
     * - sink removes particles that escaped the bounded simulation region
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
     * @brief Interactive visualization path used when Vizkit is enabled.
     */
    atlas::vizkit::Viewer<T> viewer = atlas::vizkit::Viewer<T>::builder()
                                          .with_system(system)
                                          .with_title(config::kViewerTitle)
                                          .with_size(config::kViewerWidth, config::kViewerHeight)
                                          .build();

    /**
     * @brief Fit the initial camera to the cylinder bounding box.
     */
    viewer.camera().fit_bounds(bounds.lower_corner, bounds.upper_corner);

    /**
     * @brief Render the live particle cloud.
     */
    viewer.add_layer(
        atlas::vizkit::ParticleLayer<T>::builder()
            .with_system(system)
            .with_color(config::kParticleColor)
            .make_shared());

    /**
     * @brief Render the domain bounding box as a wireframe reference.
     */
    viewer.add_layer(
        atlas::vizkit::BoxLayer<T>::builder()
            .with_unit(domain_unit)
            .make_shared());

    /**
     * @brief Render the cylinder as a semi-transparent wireframe reference.
     */
    const auto cylinder_layer = atlas::vizkit::CylinderLayer<T>::builder()
                                    .with_unit(cylinder_unit)
                                    .with_slices(config::kCylinderSlices)
                                    .make_shared();
    cylinder_layer->set_color(config::kCylinderColor);
    viewer.add_layer(cylinder_layer);

    std::cout
        << "Nitrogen DSMC example: 300 K, zero bulk drift, circular source near +z, flow inside a finite cylinder.\n";

    const int exit_code = viewer.run();
    observer->export_csv(observer_output_path());
    return exit_code;
#else
    /**
     * @brief Headless fallback path for builds without Vizkit.
     *
     * This path keeps the example runnable in environments without OpenGL or
     * GLFW while still exercising the same simulation pipeline.
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