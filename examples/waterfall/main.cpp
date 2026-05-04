#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <iostream>
#include <utility>

namespace {

using T    = float;
using Vec3 = atlas::Vector3<T>;

namespace config {

    /**
     * @brief Viewer dimensions used by the Vizkit path.
     */
    constexpr int kViewerWidth  = 1440;
    constexpr int kViewerHeight = 900;

    /**
     * @brief Core SPH simulation parameters.
     *
     * `kDt` controls the integration step size.
     * `kCellSize` defines the background spatial grid resolution used by the searcher.
     * `kSourceSpacing` controls how densely the source volume is sampled.
     * `kBufferSize` is the maximum particle capacity of the fluid container.
     */
    constexpr T kDt                   = 0.004f;
    constexpr T kCellSize             = 0.32f;
    constexpr T kSourceSpacing        = 1.0f;
    constexpr std::size_t kBufferSize = 90000;

    /**
     * @brief Material parameters used for the water-like SPH fluid.
     */
    constexpr T kWaterMass                = 1.0f;
    constexpr T kWaterRestDensity         = 1.0f;
    constexpr T kWaterPressureCoefficient = 6.5f;
    constexpr T kWaterDynamicViscosity    = 0.035f;
    constexpr T kWaterSmoothingLength     = 0.55f;

    /**
     * @brief Source velocity generator parameters.
     *
     * Emitted particles receive a jittered scalar source value centered around
     * `kSourceBaseVelocity`.
     */
    constexpr T kSourceBaseVelocity   = -2.8f;
    constexpr T kSourceVelocityJitter = 0.35f;

    /**
     * @brief Axis-aligned simulation domain bounds.
     */
    const Vec3 kDomainMin(-40.0f, -20.0f, -10.0f);
    const Vec3 kDomainMax(6.0f, 20.0f, 10.0f);

    /**
     * @brief Volume source region placed near one side of the domain.
     *
     * Particles are spawned inside this box and then evolve under SPH dynamics,
     * gravity, collision, and sink removal.
     */
    const Vec3 kSourceMin(5.5f, -14.5f, 3.2f);
    const Vec3 kSourceMax(6.0f, 14.5f, 8.4f);

    /**
     * @brief Constant gravity applied by the orchestrator.
     */
    const Vec3 kGravity(0.0f, 0.0f, -9.81f);

    /**
     * @brief Rotating tetrahedron collider and particle rendering settings.
     */
    constexpr T kParticlePointSize     = 0.5f;
    constexpr T kTetrahedronHalfExtent = 4.5f;
    const Vec3 kTetrahedronSpin(0.35f, 0.55f, 0.90f);

    /**
     * @brief Visualization colors and viewer title.
     */
    const atlas::Vector4<T> kParticleColor(0.12f, 0.70f, 0.98f, 0.92f);
    const atlas::Vector4<T> kDomainColor(0.92f, 0.96f, 0.98f, 0.45f);
    const atlas::Vector4<T> kTetrahedronColor(0.92f, 0.96f, 0.98f, 0.45f);

    constexpr auto kViewerTitle = "Atlas Waterfall SPH";

} // namespace config

/**
 * @brief Create an identity sync object for static world-space placement.
 *
 * The returned sync stores zero translation and identity orientation.
 *
 * @return Host-shared pointer to the identity sync.
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
 * The unit uses an identity pose and a constant angular velocity.
 *
 * @param geometry Geometry to attach to the unit.
 * @param angular_velocity Constant angular velocity assigned to the unit.
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
 * particles that leave the valid simulation region.
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
 * @brief Build the volume source geometry.
 *
 * Particles are spawned inside this box using volume sampling.
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
 * The tetrahedron is constructed explicitly from four vertices and four faces.
 * Face winding is corrected so triangle orientation remains consistent with the
 * outward-facing side of the tetrahedron.
 *
 * @return Host-shared pointer to the tetrahedron triangle mesh geometry.
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
 * @brief Create the SPH fluid container and configure its material and generator.
 *
 * The example uses a single water-like material entry and a jittering generator
 * used by the source.
 *
 * @return Host-shared pointer to the configured fluid.
 */
atlas::FluidHostPtr<T>
make_fluid() {
    atlas::HostBuffer<atlas::MaterialProperties<T>> material_properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    material_properties[0] = atlas::MaterialProperties<T>::builder()
                                 .with_type(atlas::MaterialType::Molecule)
                                 .with_mass(config::kWaterMass)
                                 .with_molecular_mass(config::kWaterMass)
                                 .with_rest_density(config::kWaterRestDensity)
                                 .with_pressure_coefficient(config::kWaterPressureCoefficient)
                                 .with_dynamic_viscosity(config::kWaterDynamicViscosity)
                                 .with_smoothing_length(config::kWaterSmoothingLength)
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
        .make_host_shared();
}

/**
 * @brief Create the simulation universe covering the configured domain bounds.
 *
 * @return Host-shared pointer to the configured universe.
 */
atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::Universe<T>::builder()
        .with_lower_corner(config::kDomainMin)
        .with_upper_corner(config::kDomainMax)
        .with_cell_size(config::kCellSize)
        .make_host_shared();
}

} // namespace

/**
 * @brief Entry point of the SPH waterfall example.
 *
 * The runtime pipeline consists of:
 * - a fluid container and simulation universe
 * - a spatial hashing searcher
 * - an SPH solver
 * - an orchestrator applying gravity and solver updates
 * - a volume source
 * - a domain sink
 * - a rotating tetrahedron collider
 *
 * When Vizkit is enabled, the system runs interactively. Otherwise it executes
 * a fixed number of headless simulation steps and prints a summary.
 *
 * @return Process exit code.
 */
int
main() {
    const auto fluid    = make_fluid();
    const auto universe = make_universe();

    /**
     * @brief Create the spatial searcher used for particle neighborhood lookup.
     */
    const auto searcher = atlas::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    /**
     * @brief Create the SPH solver using the cubic spline kernel.
     */
    const auto sph_solver = atlas::SphSolver<T>::builder()
                                .with_universe(universe)
                                .with_fluid(fluid)
                                .with_searcher(searcher)
                                .with_kernel_type(atlas::system::SphKernelType::cubic_spline)
                                .make_host_shared();

    /**
     * @brief Create the orchestrator that applies gravity and advances the solver.
     */
    const auto orchestrator = atlas::Orchestrator<T>::builder()
                                  .with_universe(universe)
                                  .with_fluid(fluid)
                                  .with_searcher(searcher)
                                  .with_gravity(config::kGravity)
                                  .with_solver(sph_solver)
                                  .make_host_shared();

    /**
     * @brief Build scene geometry for the domain, source, and rotating collider.
     */
    const auto domain_geometry      = make_domain_geometry();
    const auto source_geometry      = make_source_geometry();
    const auto tetrahedron_geometry = make_tetrahedron_geometry();

    /**
     * @brief Wrap geometry into runtime units.
     *
     * The tetrahedron unit is assigned a constant angular velocity so it rotates
     * during simulation and visualization.
     */
    const auto domain_unit      = make_unit(domain_geometry);
    const auto source_unit      = make_unit(source_geometry);
    const auto tetrahedron_unit = make_rotating_unit(
        tetrahedron_geometry,
        config::kTetrahedronSpin);

    /**
     * @brief Create the volume source that spawns particles inside the source box.
     *
     * The source uses volume sampling and zero temperature, so the configured
     * source generator drives the emitted particle values directly.
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
     * @brief Create the sink that removes particles outside the domain box.
     *
     * `with_flip(true)` inverts the inside/outside test so particles are kept
     * inside the domain and removed once they escape it.
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
     * @brief Create the collider using the rotating tetrahedron triangle mesh.
     */
    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *tetrahedron_unit })
                              .make_host_shared();

    /**
     * @brief Assemble the top-level simulation system.
     *
     * The per-update execution order is:
     * - source emits particles
     * - orchestrator updates SPH dynamics
     * - collider resolves tetrahedron interactions
     * - sink removes particles that left the domain
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
     * @brief Run the interactive Vizkit viewer path.
     */
    auto viewer = atlas::vizkit::Viewer<T>::builder()
                      .with_system(system)
                      .with_title(config::kViewerTitle)
                      .with_size(config::kViewerWidth, config::kViewerHeight)
                      .build();

    viewer.camera().fit_bounds(config::kDomainMin, config::kDomainMax);

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
     * @brief Render the outer domain box as a wireframe reference.
     */
    const auto domain_layer = atlas::vizkit::BoxLayer<T>::builder()
                                  .with_unit(domain_unit)
                                  .make_shared();
    domain_layer->set_color(config::kDomainColor);
    viewer.add_layer(domain_layer);

    /**
     * @brief Render the rotating tetrahedron mesh.
     */
    const auto tetrahedron_layer = atlas::vizkit::TriangleMeshLayer<T>::builder()
                                       .with_unit(tetrahedron_unit)
                                       .make_shared();
    tetrahedron_layer->set_color(config::kTetrahedronColor);
    viewer.add_layer(tetrahedron_layer);

    std::cout
        << "SPH waterfall example: rotating tetrahedron triangle-mesh collider at the domain center.\n";
    return viewer.run();
#else
    /**
     * @brief Run the headless fallback path.
     */
    for (int step = 0; step < 1200; ++step) {
        system->update();
    }

    std::cout << "SPH waterfall ran headlessly.\n"
              << "Active particles: " << fluid->particle_count() << '\n';
    return 0;
#endif
}