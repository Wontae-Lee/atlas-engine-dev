#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <iostream>
#include <utility>
#include <vector>

namespace {

using T    = float;
using Vec3 = atlas::Vector3<T>;

namespace config {

    /**
     * @brief Viewer dimensions used by the Vizkit execution path.
     */
    constexpr int kViewerWidth  = 2440;
    constexpr int kViewerHeight = 1900;

    /**
     * @brief Core simulation parameters shared by the example.
     *
     * - `kDt` controls the per-step integration interval.
     * - `kCellSize` defines the universe grid resolution used by search, measurement,
     *   codec classification, and solver execution.
     * - `kSourceSpacing` controls how densely the source volume is sampled.
     * - `kBufferSize` sets the particle capacity of the fluid container.
     * - `kHeadlessSteps` is the number of updates executed in non-visual mode.
     */
    constexpr T kDt                   = 0.003f;
    constexpr T kCellSize             = 0.50f;
    constexpr T kSourceSpacing        = 0.70f;
    constexpr std::size_t kBufferSize = 500000;
    constexpr int kHeadlessSteps      = 1200;

    /**
     * @brief Material parameters used for the single simulated particle species.
     *
     * The example mixes SPH-style solver behavior and DSMC-style solver behavior
     * through the orchestrator, so the material record provides both:
     * - SPH-related fields such as rest density, pressure coefficient,
     *   dynamic viscosity, and smoothing length
     * - DSMC-related fields such as molecular mass and collision diameter
     */
    constexpr T kParticleMass                = 1.0f;
    constexpr T kParticleMolecularMass       = 1.0f;
    constexpr T kParticleRestDensity         = 1.0f;
    constexpr T kParticlePressureCoefficient = 4.5f;
    constexpr T kParticleDynamicViscosity    = 0.025f;
    constexpr T kParticleSmoothingLength     = 0.80f;
    constexpr T kParticleCollisionDiameter   = 0.15f;
    constexpr T kStatisticalWeight           = 1.0f;

    /**
     * @brief Source generator and codec configuration parameters.
     *
     * The source uses a jittering generator centered on `kSourceBaseVelocity`.
     * The Knudsen codec uses `kCodecCharacteristicLength` to classify cells and
     * assign them to the appropriate solver family.
     */
    constexpr T kSourceBaseVelocity          = -2.4f;
    constexpr T kSourceVelocityJitter        = 5.5f;
    // Raise the characteristic length so the Knudsen split is less biased toward
    // the highest solver bucket, which keeps the red assignment closer to half.
    constexpr T kCodecCharacteristicLength   = 1.e-1f;
    constexpr int kGatewayGroupParticleCount = 6;

    /**
     * @brief Simulation domain bounds.
     */
    const Vec3 kDomainMin(-40.0f, -20.0f, -10.0f);
    const Vec3 kDomainMax(6.0f, 20.0f, 10.0f);

    /**
     * @brief Volume source region.
     *
     * New particles are spawned inside this box and then evolve under orchestrated
     * SPH/DSMC updates, gravity, collider interaction, and domain sink removal.
     */
    const Vec3 kSourceMin(5.5f, -14.5f, 3.2f);
    const Vec3 kSourceMax(6.0f, 14.5f, 8.4f);

    /**
     * @brief Constant gravity applied by the orchestrator.
     */
    const Vec3 kGravity(0.0f, 0.0f, -9.81f);

    /**
     * @brief Tetrahedron collider and particle rendering parameters.
     */
    constexpr T kParticlePointSize     = 2.5f;
    constexpr T kTetrahedronHalfExtent = 4.5f;
    const Vec3 kTetrahedronSpin(0.35f, 0.55f, 0.90f);

    /**
     * @brief Visualization colors and viewer title.
     *
     * The fallback particle color is used when a solver-specific color is not
     * available from the orchestrator layer.
     */
    const atlas::Vector4<float> kFallbackParticleColor(1.00f, 0.84f, 0.18f, 0.96f);
    const atlas::Vector4<T> kDomainColor(0.92f, 0.96f, 0.98f, 0.45f);
    const atlas::Vector4<T> kTetrahedronColor(0.92f, 0.96f, 0.98f, 0.45f);

    constexpr const char* kViewerTitle = "Atlas Codec Orchestrator";

} // namespace config

/**
 * @brief Create an identity sync object for static world-space placement.
 *
 * @return Host-shared pointer to an identity sync.
 */
atlas::SyncHostPtr<T>
make_identity_sync() {
    return atlas::Sync<T>::builder()
        .with_rigid_pose(Vec3(0, 0, 0), atlas::Quaternion<T>(1, 0, 0, 0))
        .make_host_shared();
}

/**
 * @brief Wrap a geometry object in a static unit.
 *
 * @param geometry Geometry to attach to the unit.
 * @return Host-shared pointer to the constructed unit.
 */
atlas::UnitHostPtr<T>
make_unit(const atlas::GeometryHostPtr<T>& geometry) {
    auto builder = atlas::Unit<T>::builder();
    builder.with_geometry(geometry).with_sync(make_identity_sync());
    return builder.make_host_shared();
}

/**
 * @brief Wrap a geometry object in a rotating unit.
 *
 * @param geometry Geometry to attach to the unit.
 * @param angular_velocity Constant angular velocity applied to the unit.
 * @return Host-shared pointer to the constructed rotating unit.
 */
atlas::UnitHostPtr<T>
make_rotating_unit(const atlas::GeometryHostPtr<T>& geometry,
                   const Vec3& angular_velocity) {
    auto builder = atlas::Unit<T>::builder();
    builder.with_geometry(geometry)
        .with_sync(make_identity_sync())
        .with_angular_velocity(angular_velocity);
    return builder.make_host_shared();
}

/**
 * @brief Build the outer domain box geometry.
 *
 * This geometry is used both for visualization and for sink-based removal of
 * particles that leave the simulation domain.
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
 * @brief Build the source box geometry used for volume spawning.
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
 * @brief Build a tetrahedron triangle mesh centered at the origin.
 *
 * Face winding is corrected so that all triangle normals are oriented
 * consistently outward.
 *
 * @return Host-shared pointer to the tetrahedron mesh geometry.
 */
atlas::GeometryHostPtr<T>
make_tetrahedron_geometry() {
    const T half_extent = config::kTetrahedronHalfExtent;

    const Vec3 v0(half_extent, half_extent, half_extent);
    const Vec3 v1(-half_extent, -half_extent, half_extent);
    const Vec3 v2(-half_extent, half_extent, -half_extent);
    const Vec3 v3(half_extent, -half_extent, -half_extent);

    atlas::HostBuffer<atlas::TriangleContainer4<T>> triangles(4);

    auto write_face = [&](const std::size_t face_index,
                          const Vec3& a,
                          const Vec3& b,
                          const Vec3& c,
                          const Vec3& opposite_vertex) {
        auto& triangle = triangles[face_index];
        triangle.a()   = a;
        triangle.b()   = b;
        triangle.c()   = c;

        const Vec3 face_center = (a + b + c) / static_cast<T>(3);
        const Vec3 face_normal = atlas::math::cross(triangle.b() - triangle.a(), triangle.c() - triangle.a());

        if (atlas::math::dot(face_normal, opposite_vertex - face_center) > static_cast<T>(0)) {
            std::swap(triangle.b(), triangle.c());
        }
    };

    write_face(0, v0, v1, v2, v3);
    write_face(1, v0, v3, v1, v2);
    write_face(2, v0, v2, v3, v1);
    write_face(3, v1, v3, v2, v0);

    return atlas::geometry::TriangleMesh<T>::builder()
        .with_triangles(std::move(triangles))
        .make_host_shared();
}

/**
 * @brief Create the single-species fluid used by the example.
 *
 * The material definition intentionally includes both SPH-oriented and
 * DSMC-oriented parameters so that the codec-driven orchestrator can route
 * different cells to different solver families while operating on the same
 * particle population.
 *
 * @return Host-shared pointer to the configured fluid.
 */
atlas::FluidHostPtr<T>
make_fluid() {
    atlas::HostBuffer<atlas::MaterialProperties<T>> material_properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    material_properties[0] = atlas::MaterialProperties<T>::builder()
                                 .with_type(atlas::MaterialType::Molecule)
                                 .with_mass(config::kParticleMass)
                                 .with_molecular_mass(config::kParticleMolecularMass)
                                 .with_collision_diameter(config::kParticleCollisionDiameter)
                                 .with_rest_density(config::kParticleRestDensity)
                                 .with_pressure_coefficient(config::kParticlePressureCoefficient)
                                 .with_dynamic_viscosity(config::kParticleDynamicViscosity)
                                 .with_smoothing_length(config::kParticleSmoothingLength)
                                 .with_species_id(0)
                                 .build();

    generators[0] = atlas::fluid::JitteringOperator<T>::builder()
                        .with_base_value(config::kSourceBaseVelocity)
                        .with_jitter_radius(config::kSourceVelocityJitter)
                        .with_seed(7u)
                        .make_host_shared();

    return atlas::Fluid<T>::builder()
        .with_buffer_size(config::kBufferSize)
        .with_properties(material_properties)
        .with_generators(generators)
        .with_statistical_weight(config::kStatisticalWeight)
        .make_host_shared();
}

/**
 * @brief Create the simulation universe covering the configured domain bounds.
 *
 * @return Host-shared pointer to the universe.
 */
atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::Universe<T>::builder()
        .with_lower_corner(config::kDomainMin)
        .with_upper_corner(config::kDomainMax)
        .with_cell_size(config::kCellSize)
        .make_host_shared();
}

/**
 * @brief Return the solver colors used by the orchestrator visualization layer.
 *
 * The order should match the order in which solvers are registered into the
 * orchestrator.
 *
 * @return Vector of solver-specific RGBA colors.
 */
std::vector<atlas::Vector4<float>>
make_solver_colors() {
    return {
        atlas::Vector4<float>(0.00f, 0.82f, 1.00f, 0.96f),
        atlas::Vector4<float>(1.00f, 0.22f, 0.50f, 0.96f),
    };
}

} // namespace

/**
 * @brief Entry point of the codec-driven hybrid orchestrator example.
 *
 * This example assembles a runtime pipeline consisting of:
 * - a shared particle fluid and simulation universe
 * - a spatial searcher
 * - a Boltzmann measurer
 * - a Knudsen codec for cell classification
 * - one SPH gateway solver
 * - one DSMC NTC solver
 * - a volume source, a domain sink, and a rotating tetrahedron collider
 *
 * The orchestrator uses the codec classification to decide which solver family
 * should be applied to which region of the domain, and the visualization layer
 * colors particles according to solver assignment.
 *
 * @return Process exit code.
 */
int
main() {
    const auto fluid    = make_fluid();
    const auto universe = make_universe();

    /**
     * @brief Build the shared search structure used by measurement, codec, and solvers.
     */
    const auto searcher = atlas::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    /**
     * @brief Build the measurer that populates field-level statistics on the universe grid.
     */
    const auto measurer = atlas::BoltzmanMeasurer<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_measure_mode(atlas::MeasureModeType::Field)
                              .make_host_shared();

    /**
     * @brief Build the Knudsen codec used to classify cells for solver routing.
     *
     * The characteristic length controls the Knudsen-number-based regime split
     * used by the orchestrator.
     */
    const auto codec = atlas::KnudsenCodec<T>::builder()
                           .with_domain(universe)
                           .with_fluid(fluid)
                           .with_searcher(searcher)
                           .with_characteristic_length(config::kCodecCharacteristicLength)
                           .make_host_shared();

    /**
     * @brief Build the grouped SPH solver used for continuum-like cells.
     */
    const auto sph_gateway_solver = atlas::SphGatewaySolver<T>::builder()
                                        .with_universe(universe)
                                        .with_fluid(fluid)
                                        .with_searcher(searcher)
                                        .with_kernel_type(atlas::system::SphKernelType::cubic_spline)
                                        .with_group_particle_count(config::kGatewayGroupParticleCount)
                                        .make_host_shared();

    /**
     * @brief Build the DSMC solver used for rarefied cells.
     */
    const auto dsmc_solver = atlas::DsmcNtcSolver<T>::builder()
                                 .with_universe(universe)
                                 .with_fluid(fluid)
                                 .with_searcher(searcher)
                                 .with_kernel_type(atlas::system::DsmcKernelType::hard_sphere)
                                 .make_host_shared();

    /**
     * @brief Build the top-level orchestrator.
     *
     * One SPH solver and one DSMC solver are registered here. The orchestrator
     * layer in Vizkit uses the same ordering when mapping solver colors to
     * particles.
     */
    const auto orchestrator = atlas::Orchestrator<T>::builder()
                                  .with_universe(universe)
                                  .with_fluid(fluid)
                                  .with_searcher(searcher)
                                  .with_codec(codec)
                                  .with_measurer(measurer)
                                  .with_gravity(config::kGravity)
                                  .with_solver(sph_gateway_solver)
                                  .with_solver(dsmc_solver)
                                  .make_host_shared();

    /**
     * @brief Build geometry for the outer domain, source region, and rotating collider.
     */
    const auto domain_geometry      = make_domain_geometry();
    const auto source_geometry      = make_source_geometry();
    const auto tetrahedron_geometry = make_tetrahedron_geometry();

    /**
     * @brief Wrap scene geometry into runtime units.
     */
    const auto domain_unit      = make_unit(domain_geometry);
    const auto source_unit      = make_unit(source_geometry);
    const auto tetrahedron_unit = make_rotating_unit(tetrahedron_geometry, config::kTetrahedronSpin);

    /**
     * @brief Build the volume source.
     *
     * The source uses volume spawning and zero temperature so that the configured
     * generator directly drives emitted particle values.
     */
    const auto source = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { *source_unit })
                            .with_fluid(fluid)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                                atlas::fluid::SpawnType::Volume,
                            })
                            .with_spawn_operator(
                                atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Volume))
                            .with_spacing(config::kSourceSpacing)
                            .with_temperature(0.0f)
                            .make_host_shared();

    /**
     * @brief Build the domain sink that removes particles outside the domain box.
     *
     * `with_flip(true)` keeps particles inside the box and removes particles once
     * they leave it.
     */
    const auto sink = atlas::fluid::Sink<T>::builder()
                          .with_units(atlas::HostBuffer<atlas::Unit<T>> { *domain_unit })
                          .with_fluid(fluid)
                          .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {
                              atlas::fluid::DespawnType::Volume,
                          })
                          .with_despawn_operator(
                              atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Volume))
                          .with_flip(true)
                          .make_host_shared();

    /**
     * @brief Build the rotating tetrahedron collider.
     */
    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *tetrahedron_unit })
                              .make_host_shared();

    /**
     * @brief Seed the initial codec classification using a non-empty particle set.
     *
     * Spawning particles before the system starts ensures the first visual frame
     * already has a non-empty particle population.  The search, measurement, and
     * codec passes are only needed ahead of the first Vizkit render; in headless
     * mode the orchestrator rebuilds all three on the very first system->update()
     * call, so running them here would be redundant GPU work.
     */
    source->update(config::kDt);
#ifdef ATLAS_ENABLE_VIZKIT
    searcher->build();
    measurer->measure();
    codec->update();
#endif

    /**
     * @brief Assemble the top-level runtime system.
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
     * @brief Run the interactive Vizkit path.
     */
    auto viewer = atlas::vizkit::Viewer<T>::builder()
                      .with_system(system)
                      .with_title(config::kViewerTitle)
                      .with_size(config::kViewerWidth, config::kViewerHeight)
                      .build();

    viewer.camera().fit_bounds(config::kDomainMin, config::kDomainMax);

    /**
     * @brief Render particles colored by orchestrator solver assignment.
     */
    viewer.add_layer(
        atlas::vizkit::OrchestratorLayer<T>::builder()
            .with_orchestrator(orchestrator)
            .with_default_color(config::kFallbackParticleColor)
            .with_solver_colors(make_solver_colors())
            .with_point_size(config::kParticlePointSize)
            .make_shared());

    /**
     * @brief Render the outer domain box.
     */
    const auto domain_layer = atlas::vizkit::BoxLayer<T>::builder()
                                  .with_unit(domain_unit)
                                  .make_shared();
    domain_layer->set_color(config::kDomainColor);
    viewer.add_layer(domain_layer);

    /**
     * @brief Render the rotating tetrahedron collider mesh.
     */
    const auto tetrahedron_layer = atlas::vizkit::TriangleMeshLayer<T>::builder()
                                       .with_unit(tetrahedron_unit)
                                       .make_shared();
    tetrahedron_layer->set_color(config::kTetrahedronColor);
    viewer.add_layer(tetrahedron_layer);

    std::cout
        << "Codec orchestrator example: SPH gateway + DSMC NTC with Knudsen-based solver coloring.\n";
    return viewer.run();
#else
    /**
     * @brief Run the headless fallback path.
     */
    for (int step = 0; step < config::kHeadlessSteps; ++step) {
        system->update();
    }

    std::cout
        << "Codec orchestrator example ran headlessly for " << config::kHeadlessSteps << " steps.\n"
        << "Active particles: " << fluid->particle_count() << '\n';
    return 0;
#endif
}
