#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <iostream>
#include <utility>
#include <vector>

using namespace atlas;

int
main() {
    // Use single precision for this example to keep the hybrid orchestrator
    // particle workload compact.
    using T = float;

    // -------------------------------------------------------------------------
    // 1. Material and velocity-generation configuration
    // -------------------------------------------------------------------------
    // This block defines a single material that carries both SPH-style and
    // DSMC-style parameters so the codec can route cells to different solvers.

    // Store one material species.
    HostBuffer<MaterialProperties<T>> material_properties(1);

    // Store one generator. Its index is expected to match the material/species setup.
    HostBuffer<GeneratorHostPtr<T>> generators(1);

    // Configure the hybrid material record.
    material_properties[0] = MaterialProperties<T>::builder()
                                 // Mark this species as a moving molecule-like particle.
                                 .with_type(MaterialType::Molecule)

                                 // Set mass and molecular mass for shared solver use.
                                 .with_mass(1.0f)
                                 .with_molecular_mass(1.0f)

                                 // Set the DSMC collision diameter.
                                 .with_reference_diameter(0.15f)

                                 // Set the SPH equation-of-state and viscosity parameters.
                                 .with_rest_density(1.0f)
                                 .with_pressure_coefficient(4.5f)
                                 .with_dynamic_viscosity(0.025f)
                                 .with_smoothing_length(0.80f)

                                 // Assign a species identifier. With only one species, zero is used.
                                 .with_species_id(0)

                                 // Finalize the immutable material-property object.
                                 .build();

    // Configure the source generator.
    generators[0] = fluid::JitteringOperator<T>::builder()
                        // Center generated values on a negative base velocity.
                        .with_base_value(-2.4f)

                        // Add bounded variation around the base value.
                        .with_jitter_radius(5.5f)

                        // Use a fixed seed so generated samples are reproducible.
                        .with_seed(7u)

                        // Allocate the generator in host-managed shared ownership.
                        .make_host_shared();

    // Create the fluid particle storage and attach material/generator metadata.
    const auto fluid = Fluid<T>::builder()
                           // Reserve storage for up to 500,000 particles.
                           .with_buffer_size(500000)

                           // Attach the single-species material table.
                           .with_properties(material_properties)

                           // Attach the generator table used by source injection.
                           .with_generators(generators)

                           // Set the DSMC statistical weight used by collision statistics.
                           .with_statistical_weight(1.0f)

                           // Allocate the fluid object in host-managed shared ownership.
                           .make_host_shared();

    // -------------------------------------------------------------------------
    // 2. Core simulation containers
    // -------------------------------------------------------------------------
    // This block creates the universe, searcher, measurer, codec, SPH gateway,
    // DSMC solver, and orchestrator.

    // Create the Cartesian simulation universe.
    const auto universe = Universe<T>::builder()
                              // Define the lower corner of the computational domain.
                              .with_lower_corner(Vector3F(-40.0f, -20.0f, -10.0f))

                              // Define the upper corner of the computational domain.
                              .with_upper_corner(Vector3F(6.0f, 20.0f, 10.0f))

                              // Use a uniform cell size for search, measure, codec, and solver work.
                              .with_cell_size(0.50f)

                              // Allocate the universe object in host-managed shared ownership.
                              .make_host_shared();

    // Build the shared search structure used by measurement, codec, and solvers.
    const auto searcher = SpatialHashingSearcher<T>::builder()
                              // Attach the simulation domain used to define the grid.
                              .with_universe(universe)

                              // Attach the particle container that will be indexed.
                              .with_fluid(fluid)

                              // Allocate the searcher in host-managed shared ownership.
                              .make_host_shared();

    // Build the measurer that populates field-level statistics on the universe grid.
    const auto measurer = BoltzmanMeasurer<T>::builder()
                              // Attach the universe to define measurement cells.
                              .with_universe(universe)

                              // Attach the fluid to sample particle states.
                              .with_fluid(fluid)

                              // Attach the searcher to access particles in cell order.
                              .with_searcher(searcher)

                              // Measure field quantities for codec and visualization diagnostics.
                              .with_measure_mode(MeasureModeType::Field)

                              // Allocate the measurer in host-managed shared ownership.
                              .make_host_shared();

    // Build the Knudsen codec used to classify cells for solver routing.
    const auto codec = KnudsenCodec<T>::builder()
                           // Attach the simulation domain.
                           .with_domain(universe)

                           // Attach the particle container.
                           .with_fluid(fluid)

                           // Attach the searcher used for local particle statistics.
                           .with_searcher(searcher)

                           // Set the characteristic length for Knudsen-number classification.
                           .with_characteristic_length(1.e-1f)

                           // Allocate the codec in host-managed shared ownership.
                           .make_host_shared();

    // Build the grouped SPH solver used for continuum-like cells.
    const auto sph_gateway_solver = SphGatewaySolver<T>::builder()
                                        // Attach the universe so the solver can access cell topology.
                                        .with_universe(universe)

                                        // Attach the fluid so the solver can update particle states.
                                        .with_fluid(fluid)

                                        // Attach the searcher so the solver can traverse neighbors.
                                        .with_searcher(searcher)

                                        // Use the cubic spline kernel for SPH interpolation.
                                        .with_kernel_type(system::SphKernelType::cubic_spline)

                                        // Group particles before gateway SPH processing.
                                        .with_group_particle_count(6)

                                        // Allocate the solver in host-managed shared ownership.
                                        .make_host_shared();

    // Build the DSMC solver used for rarefied cells.
    const auto dsmc_solver = make_host_shared<DsmcSolver<T>>(
        universe,
        fluid,
        searcher,
        system::DsmcKernelType::hard_sphere);

    // Build the top-level orchestrator with codec-directed solver routing.
    const auto orchestrator = Orchestrator<T>::builder()
                                  // Attach the shared universe object.
                                  .with_universe(universe)

                                  // Attach the shared fluid object.
                                  .with_fluid(fluid)

                                  // Attach the spatial searcher used before solver/measurer work.
                                  .with_searcher(searcher)

                                  // Attach the codec that classifies cells.
                                  .with_codec(codec)

                                  // Attach the measurer to collect field data during updates.
                                  .with_measurer(measurer)

                                  // Apply constant gravity.
                                  .with_gravity(Vector3F(0.0f, 0.0f, -9.81f))

                                  // Register continuum-like and rarefied solvers in visualization order.
                                  .with_solver(sph_gateway_solver)
                                  .with_solver(dsmc_solver)

                                  // Allocate the orchestrator in host-managed shared ownership.
                                  .make_host_shared();

    // -------------------------------------------------------------------------
    // 3. Geometric regions
    // -------------------------------------------------------------------------
    // This block creates the outer domain, source box, and rotating tetrahedron
    // collider mesh.

    // Build the outer domain box.
    const auto domain_geometry = geometry::Box<T>::builder()
                                     // Define the lower corner of the simulation domain.
                                     .with_lower_corner(Vector3F(-40.0f, -20.0f, -10.0f))

                                     // Define the upper corner of the simulation domain.
                                     .with_upper_corner(Vector3F(6.0f, 20.0f, 10.0f))

                                     // Allocate the geometry in host-managed shared ownership.
                                     .make_host_shared();

    // Build the volume source box.
    const auto source_geometry = geometry::Box<T>::builder()
                                     // Define the lower corner of the source region.
                                     .with_lower_corner(Vector3F(5.5f, -14.5f, 3.2f))

                                     // Define the upper corner of the source region.
                                     .with_upper_corner(Vector3F(6.0f, 14.5f, 8.4f))

                                     // Allocate the geometry in host-managed shared ownership.
                                     .make_host_shared();

    // Build a tetrahedron triangle mesh centered at the origin.
    const T half_extent = 4.5f;
    const Vector3F v0(half_extent, half_extent, half_extent);
    const Vector3F v1(-half_extent, -half_extent, half_extent);
    const Vector3F v2(-half_extent, half_extent, -half_extent);
    const Vector3F v3(half_extent, -half_extent, -half_extent);

    HostBuffer<TriangleContainer4<T>> triangles(4);
    auto write_face = [&](const std::size_t face_index,
                          const Vector3F& a,
                          const Vector3F& b,
                          const Vector3F& c,
                          const Vector3F& opposite_vertex) {
        auto& triangle = triangles[face_index];
        triangle.a()   = a;
        triangle.b()   = b;
        triangle.c()   = c;

        const Vector3F face_center = (a + b + c) / static_cast<T>(3);
        const Vector3F face_normal = math::cross(triangle.b() - triangle.a(), triangle.c() - triangle.a());

        if (math::dot(face_normal, opposite_vertex - face_center) > static_cast<T>(0)) {
            std::swap(triangle.b(), triangle.c());
        }
    };

    write_face(0, v0, v1, v2, v3);
    write_face(1, v0, v3, v1, v2);
    write_face(2, v0, v2, v3, v1);
    write_face(3, v1, v3, v2, v0);

    const auto tetrahedron_geometry = geometry::TriangleMesh<T>::builder()
                                          // Attach the generated tetrahedron faces.
                                          .with_triangles(std::move(triangles))

                                          // Allocate the geometry in host-managed shared ownership.
                                          .make_host_shared();

    // -------------------------------------------------------------------------
    // 4. Shared transform state and units
    // -------------------------------------------------------------------------
    // Geometry objects are wrapped in units. A unit combines geometry with a
    // synchronization object that stores its rigid pose.

    const auto sync = Sync<T>::builder()
                          // Place static geometries at the origin with identity rotation.
                          .with_rigid_pose(Vector3F(0, 0, 0), Quaternion<T>(1, 0, 0, 0))

                          // Allocate the sync object in host-managed shared ownership.
                          .make_host_shared();

    const auto domain_unit = Unit<T>::builder()
                                 // Attach the full-domain geometry.
                                 .with_geometry(domain_geometry)

                                 // Attach the shared identity transform.
                                 .with_sync(sync)

                                 // Allocate the unit in host-managed shared ownership.
                                 .make_host_shared();

    const auto source_unit = Unit<T>::builder()
                                 // Attach the volume source geometry.
                                 .with_geometry(source_geometry)

                                 // Attach the shared identity transform.
                                 .with_sync(sync)

                                 // Allocate the unit in host-managed shared ownership.
                                 .make_host_shared();

    const auto tetrahedron_unit = Unit<T>::builder()
                                      // Attach the tetrahedron collider geometry.
                                      .with_geometry(tetrahedron_geometry)

                                      // Attach the shared identity transform.
                                      .with_sync(sync)

                                      // Rotate the collider during simulation.
                                      .with_angular_velocity(Vector3F(0.35f, 0.55f, 0.90f))

                                      // Allocate the unit in host-managed shared ownership.
                                      .make_host_shared();

    // -------------------------------------------------------------------------
    // 5. Boundary and interaction systems
    // -------------------------------------------------------------------------
    // This block defines volume spawning, domain removal, and collider handling.

    // Build the volume source.
    const auto source = fluid::Source<T>::builder()
                            // Use the source unit as the injection region.
                            .with_units(HostBuffer<Unit<T>> { *source_unit })

                            // Attach the particle container that receives new particles.
                            .with_fluid(fluid)

                            // Spawn particles throughout the volume of the source geometry.
                            .with_spawn_types(HostBuffer<fluid::SpawnType> {
                                fluid::SpawnType::Volume,
                            })

                            // Use a volume spawn operator matching the selected spawn type.
                            .with_spawn_operator(fluid::SpawnOperator<T>(fluid::SpawnType::Volume))

                            // Set the approximate particle spacing inside the source region.
                            .with_spacing(0.70f)

                            // Let the configured generator directly drive emitted values.
                            .with_temperature(0.0f)

                            // Allocate the source in host-managed shared ownership.
                            .make_host_shared();

    // Build the domain sink that removes particles outside the domain box.
    const auto sink = fluid::Sink<T>::builder()
                          // Use the domain unit as the sink reference region.
                          .with_units(HostBuffer<Unit<T>> { *domain_unit })

                          // Attach the particle container from which particles are removed.
                          .with_fluid(fluid)

                          // Evaluate despawning using the volume of the domain geometry.
                          .with_despawn_types(HostBuffer<fluid::DespawnType> {
                              fluid::DespawnType::Volume,
                          })

                          // Use a volume despawn operator matching the selected despawn type.
                          .with_despawn_operator(fluid::DespawnOperator<T>(fluid::DespawnType::Volume))

                          // Flip the volume test so particles outside the domain are removed.
                          .with_flip(true)

                          // Allocate the sink in host-managed shared ownership.
                          .make_host_shared();

    // Build the rotating tetrahedron collider.
    const auto collider = Collider<T>::builder()
                              // Attach the fluid whose particles will be tested against the collider.
                              .with_fluid(fluid)

                              // Use the tetrahedron unit as the solid obstacle.
                              .with_units(HostBuffer<Unit<T>> { *tetrahedron_unit })

                              // Allocate the collider in host-managed shared ownership.
                              .make_host_shared();

    // Seed particles before the first frame so codec visualization starts non-empty.
    source->update(0.003f);
#ifdef ATLAS_ENABLE_VIZKIT
    searcher->build();
    measurer->measure();
    codec->update();
#endif

    // -------------------------------------------------------------------------
    // 6. Full system assembly
    // -------------------------------------------------------------------------
    // The System object owns the high-level update sequence. It receives the
    // fluid, domain, source/sink/collider systems, solver pipeline, and timestep.

    const auto system = System<T>::builder()
                            // Attach the particle container.
                            .with_fluid(fluid)

                            // Attach the computational domain.
                            .with_domain(universe)

                            // Attach particle injection.
                            .with_source(source)

                            // Attach particle removal.
                            .with_sink(sink)

                            // Attach particle-surface collision handling.
                            .with_collider(collider)

                            // Attach the codec-directed orchestrator.
                            .with_solver(orchestrator)

                            // Set the simulation timestep in seconds.
                            .with_dt(0.003f)

                            // Allocate the full simulation system in host-managed shared ownership.
                            .make_host_shared();

#ifdef ATLAS_ENABLE_VIZKIT
    // -------------------------------------------------------------------------
    // 7. Interactive visualization
    // -------------------------------------------------------------------------
    // Run the interactive Vizkit path when visualization support is enabled.

    auto viewer = vizkit::Viewer<T>::builder()
                      // Attach the full simulation system.
                      .with_system(system)

                      // Configure the viewer window.
                      .with_title("Atlas Codec Orchestrator")
                      .with_size(2440, 1900)

                      // Build the viewer.
                      .build();

    // Frame the initial camera around the domain.
    viewer.camera().fit_bounds(Vector3F(-40.0f, -20.0f, -10.0f), Vector3F(6.0f, 20.0f, 10.0f));

    // Render particles colored by orchestrator solver assignment.
    viewer.add_layer(
        vizkit::OrchestratorLayer<T>::builder()
            .with_orchestrator(orchestrator)
            .with_default_color(Vector4<float>(1.00f, 0.84f, 0.18f, 0.96f))
            .with_solver_colors(std::vector<Vector4<float>> {
                Vector4<float>(0.00f, 0.82f, 1.00f, 0.96f),
                Vector4<float>(1.00f, 0.22f, 0.50f, 0.96f),
            })
            .with_point_size(2.5f)
            .make_shared());

    // Render the outer domain box.
    const auto domain_layer = vizkit::BoxLayer<T>::builder()
                                  .with_unit(domain_unit)
                                  .make_shared();
    domain_layer->set_color(Vector4<T>(0.92f, 0.96f, 0.98f, 0.45f));
    viewer.add_layer(domain_layer);

    // Render the rotating tetrahedron collider mesh.
    const auto tetrahedron_layer = vizkit::TriangleMeshLayer<T>::builder()
                                       .with_unit(tetrahedron_unit)
                                       .make_shared();
    tetrahedron_layer->set_color(Vector4<T>(0.92f, 0.96f, 0.98f, 0.45f));
    viewer.add_layer(tetrahedron_layer);

    std::cout
        << "Codec orchestrator example: SPH gateway + DSMC NTC with Knudsen-based solver coloring.\n";
    return viewer.run();
#else
    // -------------------------------------------------------------------------
    // 7. Headless time integration
    // -------------------------------------------------------------------------
    // Run a fixed number of update steps when Vizkit is not enabled.

    for (int step = 0; step < 1200; ++step) {
        // Advance the simulation by one timestep.
        system->update();
    }

    // Print a compact completion message and the number of active particles left
    // in the fluid container after all update steps.
    std::cout
        << "Codec orchestrator example ran headlessly for " << 1200 << " steps.\n"
        << "Active particles: " << fluid->particle_count() << '\n';
    return 0;
#endif
}
