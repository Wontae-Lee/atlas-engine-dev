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
     * @brief Gas and time-integration parameters shared by the entire example.
     *
     * This scenario models nitrogen gas near room temperature without any imposed
     * bulk drift velocity. Newly emitted particles are sampled from a thermal
     * distribution and subsequently evolve through:
     * - DSMC particle-particle collisions
     * - collisions against the central cylinder
     * - sink removal once they leave the simulation domain
     */
    constexpr T kTemperature           = 300.0f;
    constexpr T kDt                    = 2.5e-5f;
    constexpr std::size_t kBufferSize  = 200000;
    constexpr T kNitrogenMolecularMass = 4.651734e-26f;
    constexpr T kNitrogenDiameter      = 4.17e-10f;

    /**
     * @brief Spatial discretization and source emission density controls.
     *
     * `kCellSize` defines the background Cartesian grid resolution used by both
     * the spatial hashing searcher and the DSMC collision solver.
     *
     * `kSourceSpacing` controls how densely the source volume is sampled for
     * candidate emission positions. A smaller spacing increases the number of
     * particles injected per update, producing a denser visible flow at a higher
     * computational cost.
     */
    constexpr T kCellSize      = 0.25f;
    constexpr T kSourceSpacing = 0.18f;

    /**
     * @brief Viewer presentation settings used by the Vizkit path.
     */
    constexpr int kViewerWidth    = 1440;
    constexpr int kViewerHeight   = 900;
    constexpr int kCylinderSlices = 72;

    /**
     * @brief Headless execution and observer export capacity settings.
     *
     * `kHeadlessSteps` determines how many simulation updates are executed when
     * Vizkit is not available.
     *
     * `kObserverReserveCount` pre-reserves space for observer-side sensor metrics
     * to reduce reallocations during the run.
     */
    constexpr int kHeadlessSteps                = 1000;
    constexpr std::size_t kObserverReserveCount = 4096;

    /**
     * @brief Axis-aligned outer simulation domain.
     *
     * The domain is intentionally shallow along the y-axis relative to x so the
     * flow reads visually as a channel around the cylinder. It remains tall enough
     * along z to allow particles to redistribute after diffuse reflection instead
     * of immediately leaving the valid region.
     */
    const Vec3 kDomainMin(-6.0f, -2.5f, -1.2f);
    const Vec3 kDomainMax(6.0f, 2.5f, 1.2f);

    /**
     * @brief Finite source volume placed near the left side of the domain.
     *
     * A volumetric source is used instead of an infinitesimally thin inlet plane
     * so particles are visible immediately after spawning and the inlet remains
     * visually stable even when the camera is zoomed in.
     */
    const Vec3 kSourceMin(-5.1f, -0.9f, -0.18f);
    const Vec3 kSourceMax(-4.0f, 0.9f, 0.18f);

    /**
     * @brief Geometry parameters of the central cylindrical obstacle.
     */
    const Vec3 kCylinderCenter(0, 0, 0);
    constexpr T kCylinderRadius = 0.9f;
    constexpr T kCylinderHeight = 2.2f;

    /**
     * @brief Surface interaction parameters used by the cylinder collider.
     *
     * The example uses a fully diffuse reflection model:
     * - cosine-weighted hemisphere sampling for outgoing direction generation
     * - TMAC = 1, so the diffuse branch is always selected
     * - restitution = 1, so the reflected particle preserves its incident speed
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
     * @brief Measurement mode used when computing macroscopic fields.
     *
     * `Field` stores the measured quantities on the universe grid. This is
     * sufficient for the example because the measured result is used only as a
     * cell-wise diagnostic and is not fed back into per-particle temperature or
     * state.
     */
    constexpr atlas::MeasureModeType kMeasureMode = atlas::MeasureModeType::Field;

    /**
     * @brief Simple visual styling constants for the particle layer and viewer.
     */
    const atlas::Vector4<T> kParticleColor(0.10f, 0.74f, 0.92f, 0.80f);
    constexpr const char* kViewerTitle = "Atlas DSMC Nitrogen Cylinder";

} // namespace config

/**
 * @brief Return the directory path used for observer CSV export.
 *
 * The output directory is resolved relative to the current source file so the
 * example writes its diagnostic data next to the example source tree.
 *
 * @return Filesystem path to the observer output directory.
 */
std::filesystem::path
observer_output_path() {
    return std::filesystem::path(__FILE__).parent_path() / "observer_output";
}

/**
 * @brief Create a static world-space unit for the given geometry.
 *
 * In this example, all units are fixed rigid objects. The geometry provides the
 * analytic shape, while the sync object supplies the rigid transform consumed by
 * collider queries and Vizkit visualization layers.
 *
 * @param geometry Geometry instance to wrap inside a unit.
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
 * @brief Create the fluid container and configure its species and generators.
 *
 * Atlas fluids own:
 * - material and species properties
 * - host-configured generator models
 * - fixed-capacity particle state buffers
 *
 * This example contains exactly one species, nitrogen, so both the material
 * property table and the generator table contain a single entry.
 *
 * @param observer Observer used to record runtime metrics.
 * @return Host-shared pointer to the configured fluid.
 */
atlas::FluidHostPtr<T>
make_fluid(const atlas::ObserverHostPtr& observer) {
    atlas::HostBuffer<atlas::MatrialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    /**
     * @brief Configure the nitrogen material record consumed by the DSMC solver.
     *
     * The hard-sphere collision kernel reads at least:
     * - molecular_mass
     * - collision_diameter
     *
     * A species identifier is also installed so the runtime retains an explicit
     * species label for the particle population.
     */
    properties[0] = atlas::MatrialProperties<T>::builder()
                        .with_type(atlas::MaterialType::Molecule)
                        .with_mass(config::kNitrogenMolecularMass)
                        .with_molecular_mass(config::kNitrogenMolecularMass)
                        .with_species_id(0)
                        .with_collision_diameter(config::kNitrogenDiameter)
                        .build();

    /**
     * @brief Configure the thermal source generator for newly spawned particles.
     *
     * A Maxwell-Boltzmann velocity generator is used. The bulk velocity is zero,
     * so no directed streamwise drift is prescribed at emission time. All visible
     * motion therefore emerges from thermal injection followed by DSMC and wall
     * interactions.
     */
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
        .with_observer(observer)
        .make_host_shared();
}

/**
 * @brief Build the geometry representing the outer simulation domain.
 *
 * The domain geometry serves two independent purposes:
 * - it can be rendered as a wireframe bounding box in Vizkit
 * - it is used by the sink as the reference volume for deciding whether a
 *   particle remains inside the valid simulation region
 *
 * @return Host-shared pointer to the domain box geometry.
 */
atlas::GeometryHostPtr<T>
make_domain_geometry() {
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(config::kDomainMin)
        .with_upper_corner(config::kDomainMax)
        .make_host_shared();
}

/**
 * @brief Build the source geometry used for volumetric particle emission.
 *
 * The source is a compact box near the left side of the domain and continuously
 * injects thermally sampled particles into the channel.
 *
 * @return Host-shared pointer to the source box geometry.
 */
atlas::GeometryHostPtr<T>
make_source_geometry() {
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(config::kSourceMin)
        .with_upper_corner(config::kSourceMax)
        .make_host_shared();
}

/**
 * @brief Build the geometry of the central cylindrical obstacle.
 *
 * The cylinder is centered at the origin and aligned with the z-axis.
 *
 * @return Host-shared pointer to the cylinder geometry.
 */
atlas::GeometryHostPtr<T>
make_cylinder_geometry() {
    return atlas::geometry::Cylinder<T>::builder()
        .with_center(config::kCylinderCenter)
        .with_radius(config::kCylinderRadius)
        .with_height(config::kCylinderHeight)
        .make_host_shared();
}

/**
 * @brief Construct the wall interaction law used by the cylinder collider.
 *
 * Diffuse reflection randomizes the outgoing direction according to the selected
 * sampling model. The reflected speed magnitude is still controlled separately by
 * the restitution parameter inside `ColliderSurfaceInteraction`.
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
 * @brief Entry point of the DSMC nitrogen-cylinder example.
 *
 * The program constructs the full simulation pipeline, including:
 * - observer and particle storage
 * - universe grid and spatial searcher
 * - DSMC collision solver and macroscopic field measurer
 * - source, sink, and cylinder collider
 * - optional Vizkit visualization
 *
 * When Vizkit is enabled, the example runs interactively. Otherwise it falls
 * back to a headless execution path that advances the same runtime system for a
 * fixed number of steps and prints a summary to stdout.
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
     * @brief Create the particle container first.
     *
     * Most runtime subsystems depend directly or indirectly on the fluid, so it is
     * constructed before the universe, searcher, solver, and system wrapper.
     */
    const auto fluid = make_fluid(observer);

    /**
     * @brief Create the universe that defines the simulation extents and grid.
     *
     * The regular cell grid owned by the universe is reused by both the spatial
     * hashing searcher and the DSMC collision solver.
     */
    const auto universe = atlas::Universe<T>::builder()
                              .with_lower_corner(config::kDomainMin)
                              .with_upper_corner(config::kDomainMax)
                              .with_cell_size(config::kCellSize)
                              .with_observer(observer)
                              .make_host_shared();

    /**
     * @brief Create the spatial hashing searcher.
     *
     * Each update maps active particles into grid cells so the DSMC solver can
     * query local collision neighborhoods efficiently.
     */
    const auto searcher = atlas::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    /**
     * @brief Build the DSMC solver for particle-particle collisions.
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
     * After particles are binned by the searcher, the measurer derives cell-wise
     * diagnostic quantities such as bulk velocity and temperature.
     */
    const auto measurer = atlas::BoltzmanMeasurer<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_measure_mode(config::kMeasureMode)
                              .make_host_shared();

    /**
     * @brief Build the orchestrator for the internal search/measure/solve stage.
     *
     * The orchestrator owns the runtime sequence:
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
     * @brief Construct the three scene geometries:
     * - outer domain box
     * - left-side source volume
     * - centered cylinder obstacle
     */
    const auto domain_geometry   = make_domain_geometry();
    const auto source_geometry   = make_source_geometry();
    const auto cylinder_geometry = make_cylinder_geometry();

    /**
     * @brief Wrap each geometry in a unit.
     *
     * The same units are shared between runtime systems and Vizkit rendering
     * layers so that simulation objects and visual objects remain aligned.
     */
    const auto domain_unit   = make_unit(domain_geometry);
    const auto source_unit   = make_unit(source_geometry);
    const auto cylinder_unit = make_unit(cylinder_geometry);

    /**
     * @brief Create the volumetric source that injects thermal particles.
     *
     * No streamwise bulk velocity is imposed at the inlet. The resulting flow
     * pattern emerges from repeated thermal emission combined with particle
     * collisions and scattering at the cylindrical obstacle.
     */
    const auto source = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { *source_unit })
                            .with_fluid(fluid)
                            .with_observer(observer)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                                atlas::fluid::SpawnType::Volume,
                            })
                            .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Volume))
                            .with_spacing(config::kSourceSpacing)
                            .with_temperature(config::kTemperature)
                            .make_host_shared();

    /**
     * @brief Create the sink that removes particles outside the simulation domain.
     *
     * The sink evaluates particle positions against the domain volume and uses
     * `flip=true`, which inverts the default inclusion behavior:
     * - inside the domain  -> keep the particle
     * - outside the domain -> remove the particle
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
     * @brief Create the collider that applies diffuse wall reflection on the cylinder.
     */
    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *cylinder_unit })
                              .with_surface_interactions(
                                  atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> {
                                      make_collider_interaction(),
                                  })
                              .make_host_shared();

    /**
     * @brief Assemble the top-level runtime system.
     *
     * The per-update execution order is:
     * - source emits particles
     * - orchestrator performs search, measurement, and DSMC collision solving
     * - collider resolves wall interactions
     * - sink removes particles that escaped the domain
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
     * @brief Interactive visualization path enabled when Vizkit is available.
     */
    atlas::vizkit::Viewer<T> viewer = atlas::vizkit::Viewer<T>::builder()
                                          .with_system(system)
                                          .with_title(config::kViewerTitle)
                                          .with_size(config::kViewerWidth, config::kViewerHeight)
                                          .build();

    /**
     * @brief Initialize the camera to frame the full simulation domain.
     */
    viewer.camera().fit_bounds(config::kDomainMin, config::kDomainMax);

    /**
     * @brief Add the live particle cloud rendering layer.
     */
    viewer.add_layer(
        atlas::vizkit::ParticleLayer<T>::builder()
            .with_system(system)
            .with_color(config::kParticleColor)
            .make_shared());

    /**
     * @brief Add a wireframe visualization of the outer domain bounds.
     */
    viewer.add_layer(
        atlas::vizkit::BoxLayer<T>::builder()
            .with_unit(domain_unit)
            .make_shared());

    /**
     * @brief Add a wireframe visualization of the cylinder obstacle.
     */
    viewer.add_layer(
        atlas::vizkit::CylinderLayer<T>::builder()
            .with_unit(cylinder_unit)
            .with_slices(config::kCylinderSlices)
            .make_shared());

    std::cout
        << "Nitrogen DSMC example: 300 K, zero bulk velocity, inlet-only source, outlet sinks, centered cylinder.\n";

    const int exit_code = viewer.run();
    observer->export_csv(observer_output_path());
    return exit_code;
#else
    /**
     * @brief Headless fallback path for builds without Vizkit.
     *
     * This path preserves the same simulation pipeline while avoiding any OpenGL
     * or GLFW dependency. It is useful for CI, servers, and non-graphical build
     * environments.
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