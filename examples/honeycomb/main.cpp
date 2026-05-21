#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace atlas;

int
main() {
    // Use single precision for this example to reduce memory traffic and improve
    // throughput in particle-heavy DSMC-style simulations.
    using T = float;

    // -------------------------------------------------------------------------
    // 1. Observer, material, and velocity-generation configuration
    // -------------------------------------------------------------------------
    // This block defines the runtime observer, nitrogen material properties, and
    // the Maxwell-Boltzmann generator used by source emission.

    // Create an observer with pre-reserved metric storage for source/sink data.
    const auto observer = Observer::builder()
                              .with_source_sensor_matrics(4096)
                              .with_sink_sensor_matrics(4096)
                              .make_host_shared();

    // Store one material species. The current example models a single molecular gas.
    HostBuffer<MaterialProperties<T>> material_properties(1);

    // Store one generator. Its index is expected to match the material/species setup.
    HostBuffer<GeneratorHostPtr<T>> generators(1);

    // Configure the nitrogen molecule properties.
    material_properties[0] = MaterialProperties<T>::builder()
                                 // Mark this species as a molecule rather than a wall,
                                 // marker, or other possible material category.
                                 .with_type(MaterialType::Molecule)

                                 // Set the particle mass in kilograms.
                                 .with_mass(4.651734e-26f)

                                 // Set the molecular mass used by thermal velocity generation.
                                 .with_molecular_mass(4.651734e-26f)

                                 // Assign a species identifier. With only one species, zero is used.
                                 .with_species_id(0)

                                 // Set the reference collision diameter used by hard-sphere models.
                                 .with_reference_diameter(4.17e-10f)

                                 // Finalize the immutable material-property object.
                                 .build();

    // Configure the velocity generator for newly spawned particles.
    generators[0] = fluid::MaxwellBoltzmannGenerator<T>::builder()
                        // Use 300 K as the thermal temperature of the injected gas.
                        .with_temperature(300.0f)

                        // Use the same molecular mass as the configured material species.
                        .with_molecular_mass(4.651734e-26f)

                        // Add a macroscopic drift velocity along +y.
                        .with_bulk_velocity(Vector3F(0.0f, 1000.0f, 0.0f))

                        // Use a fixed seed so generated velocity samples are reproducible.
                        .with_seed(42u)

                        // Allocate the generator in host-managed shared ownership.
                        .make_host_shared();

    // Create the fluid particle storage and attach material/generator metadata.
    const auto fluid = Fluid<T>::builder()
                           // Reserve storage for up to 1,000,000 particles.
                           .with_buffer_size(1000000)

                           // Attach the single-species material table.
                           .with_properties(material_properties)

                           // Attach the velocity generator table used by source injection.
                           .with_generators(generators)

                           // Attach the observer used by runtime metrics.
                           .with_observer(observer)

                           // Allocate the fluid object in host-managed shared ownership.
                           .make_host_shared();

    // -------------------------------------------------------------------------
    // 2. Imported mesh and derived regions
    // -------------------------------------------------------------------------
    // This block loads the honeycomb OBJ asset, recenters the mesh, rebuilds
    // triangle normals, derives a padded domain, and creates the circular source.

    // Resolve the honeycomb OBJ path relative to this source file.
    const auto honeycomb_mesh_path = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()
                                     / "assets"
                                     / "honeycomb"
                                     / "honeycomb.obj";

    // Load the raw triangle mesh from disk.
    auto mesh = geometry::TriangleMesh<T>::builder()
                    .load_from_obj(honeycomb_mesh_path.string())
                    .build();

    // Compute the raw mesh bounds before recentering.
    spatial::AxisAlignedBoundingBox<T> raw_bounds;
    for (const auto& triangle : mesh.triangles) {
        raw_bounds.merge(triangle.a());
        raw_bounds.merge(triangle.b());
        raw_bounds.merge(triangle.c());
    }

    // Recenter the mesh around the origin and discard degenerate triangles.
    const Vector3F raw_center = raw_bounds.center();
    HostBuffer<TriangleContainer4<T>> processed_triangles;
    for (const auto& triangle : mesh.triangles) {
        TriangleContainer4<T> transformed_triangle = triangle;
        transformed_triangle.a()                   = (triangle.a() - raw_center) * 1.0f;
        transformed_triangle.b()                   = (triangle.b() - raw_center) * 1.0f;
        transformed_triangle.c()                   = (triangle.c() - raw_center) * 1.0f;

        Vector3F geometric_normal = math::cross(
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

    // Store the processed triangles in the mesh geometry.
    mesh.set_triangles(processed_triangles);
    const auto honeycomb_geometry = make_host_shared<geometry::TriangleMesh<T>>(std::move(mesh));

    // Build a padded box domain from the honeycomb mesh bounds.
    const auto honeycomb_bounds = honeycomb_geometry->bound();
    const Vector3F domain_center = honeycomb_bounds.center();
    const Vector3F domain_half_extents = honeycomb_bounds.extents() * (1.5f * 0.5f);
    const auto domain_geometry = geometry::Box<T>::builder()
                                     .with_lower_corner(domain_center - domain_half_extents)
                                     .with_upper_corner(domain_center + domain_half_extents)
                                     .make_host_shared();

    // Build a circular source near the lower y boundary of the padded domain.
    const auto domain_bounds = domain_geometry->bound();
    const Vector3F domain_extents = domain_bounds.extents();
    const T source_radius = std::max(
        T(0.05),
        T(0.5) * std::min(domain_extents.x, domain_extents.z) * T(0.95));
    const auto source_geometry = geometry::Circle<T>::builder()
                                     .with_center(Vector3F(0.0f, domain_bounds.lower_corner.y + 0.10f, 0.0f))
                                     .with_normal(Vector3F(0.0f, 1.0f, 0.0f))
                                     .with_radius(source_radius)
                                     .make_host_shared();

    // -------------------------------------------------------------------------
    // 3. Core simulation containers
    // -------------------------------------------------------------------------
    // This block creates the universe, spatial searcher, DSMC solver, field
    // measurer, and orchestrator.

    // Create the Cartesian simulation universe from the padded domain bounds.
    const auto universe = Universe<T>::builder()
                              // Define the lower corner of the computational domain.
                              .with_lower_corner(domain_bounds.lower_corner)

                              // Define the upper corner of the computational domain.
                              .with_upper_corner(domain_bounds.upper_corner)

                              // Use a uniform cell size for spatial hashing and cell-wise DSMC.
                              .with_cell_size(0.25f)

                              // Attach the observer used by runtime metrics.
                              .with_observer(observer)

                              // Allocate the universe object in host-managed shared ownership.
                              .make_host_shared();

    // Build a spatial hashing searcher that maps particles to universe cells.
    const auto searcher = SpatialHashingSearcher<T>::builder()
                              // Attach the simulation domain used to define the grid.
                              .with_universe(universe)

                              // Attach the particle container that will be indexed.
                              .with_fluid(fluid)

                              // Allocate the searcher in host-managed shared ownership.
                              .make_host_shared();

    // Configure the DSMC collision solver.
    const auto dsmc_solver = DsmcCellSequentialSolver<T>::builder()
                                 // Attach the universe so the solver can access cell topology.
                                 .with_universe(universe)

                                 // Attach the fluid so the solver can update particle velocities.
                                 .with_fluid(fluid)

                                 // Attach the searcher so the solver can traverse particles by cell.
                                 .with_searcher(searcher)

                                 // Use the hard-sphere collision kernel for molecule collisions.
                                 .with_kernel_type(system::DsmcKernelType::hard_sphere)

                                 // Allocate the solver in host-managed shared ownership.
                                 .make_host_shared();

    // Configure the field measurer.
    const auto measurer = BoltzmanMeasurer<T>::builder()
                              // Attach the universe to define measurement cells.
                              .with_universe(universe)

                              // Attach the fluid to sample particle states.
                              .with_fluid(fluid)

                              // Attach the searcher to access particles in cell order.
                              .with_searcher(searcher)

                              // Measure field quantities rather than only global quantities.
                              .with_measure_mode(MeasureModeType::Field)

                              // Allocate the measurer in host-managed shared ownership.
                              .make_host_shared();

    // Combine search, collision solving, and measurement into one solver pipeline.
    const auto orchestrator = Orchestrator<T>::builder()
                                  // Attach the shared universe object.
                                  .with_universe(universe)

                                  // Attach the shared fluid object.
                                  .with_fluid(fluid)

                                  // Attach the spatial searcher used before solver/measurer work.
                                  .with_searcher(searcher)

                                  // Attach the measurer to collect field data during updates.
                                  .with_measurer(measurer)

                                  // Attach the DSMC solver that performs collision updates.
                                  .with_solver(dsmc_solver)

                                  // Allocate the orchestrator in host-managed shared ownership.
                                  .make_host_shared();

    // -------------------------------------------------------------------------
    // 4. Shared transform state and units
    // -------------------------------------------------------------------------
    // Geometry objects are wrapped in units. A unit combines geometry with a
    // synchronization object that stores its rigid pose.

    const auto sync = Sync<T>::builder()
                          // Place the geometry at the origin with identity rotation.
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
                                 // Attach the circular source geometry.
                                 .with_geometry(source_geometry)

                                 // Attach the shared identity transform.
                                 .with_sync(sync)

                                 // Allocate the unit in host-managed shared ownership.
                                 .make_host_shared();

    const auto honeycomb_unit = Unit<T>::builder()
                                    // Attach the imported honeycomb mesh geometry.
                                    .with_geometry(honeycomb_geometry)

                                    // Attach the shared identity transform.
                                    .with_sync(sync)

                                    // Allocate the unit in host-managed shared ownership.
                                    .make_host_shared();

    // -------------------------------------------------------------------------
    // 5. Boundary and interaction systems
    // -------------------------------------------------------------------------
    // This block defines how particles enter, leave, and interact with geometry.

    // Configure circular surface particle injection.
    const auto source = fluid::Source<T>::builder()
                            // Use the source unit as the injection region.
                            .with_units(HostBuffer<Unit<T>> { *source_unit })

                            // Attach the particle container that receives new particles.
                            .with_fluid(fluid)

                            // Attach the observer used by runtime metrics.
                            .with_observer(observer)

                            // Spawn particles on the surface of the source geometry.
                            .with_spawn_types(HostBuffer<fluid::SpawnType> {
                                fluid::SpawnType::Surface,
                            })

                            // Use a surface spawn operator matching the selected spawn type.
                            .with_spawn_operator(fluid::SpawnOperator<T>(fluid::SpawnType::Surface))

                            // Set the approximate particle spacing on the source disk.
                            .with_spacing(0.1f)

                            // Use the same thermal temperature as the velocity generator.
                            .with_temperature(300.0f)

                            // Allocate the source in host-managed shared ownership.
                            .make_host_shared();

    // Configure particle removal at the padded domain boundary.
    const auto sink = fluid::Sink<T>::builder()
                          // Use the padded domain unit as the sink reference region.
                          .with_units(HostBuffer<Unit<T>> { *domain_unit })

                          // Attach the particle container from which particles are removed.
                          .with_fluid(fluid)

                          // Attach the observer used by runtime metrics.
                          .with_observer(observer)

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

    // Configure particle interaction with the honeycomb triangle mesh.
    const auto collider = Collider<T>::builder()
                              // Attach the fluid whose particles will be tested against the collider.
                              .with_fluid(fluid)

                              // Use the honeycomb unit as the solid obstacle.
                              .with_units(HostBuffer<Unit<T>> { *honeycomb_unit })

                              // Define one surface-interaction model for the honeycomb mesh.
                              .with_surface_interactions(
                                  HostBuffer<system::ColliderSurfaceInteraction<T>> {
                                      system::ColliderSurfaceInteraction<T>::builder()
                                          // Use cosine-weighted diffuse reflection.
                                          .with_diffuse_sampling(system::DiffuseSampling::CosineWeighted)

                                          // Preserve incident speed magnitude.
                                          .with_restitution(1.0f)

                                          // Use full tangential momentum accommodation.
                                          .with_tangential_momentum_accommodation(1.0f)

                                          // Store the wall temperature used by the interaction model.
                                          .with_temperature(300.0f)

                                          // Finalize the surface-interaction object.
                                          .build(),
                                  })

                              // Invert collider sidedness for the imported mesh orientation.
                              .with_flip(true)

                              // Allocate the collider in host-managed shared ownership.
                              .make_host_shared();

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

                            // Attach the orchestrated DSMC and measurement pipeline.
                            .with_solver(orchestrator)

                            // Set the simulation timestep in seconds.
                            .with_dt(2.5e-6f)

                            // Allocate the full simulation system in host-managed shared ownership.
                            .make_host_shared();

#ifdef ATLAS_ENABLE_VIZKIT
    // -------------------------------------------------------------------------
    // 7. Interactive visualization
    // -------------------------------------------------------------------------
    // Run the interactive Vizkit path when visualization support is enabled.

    vizkit::Viewer<T> viewer = vizkit::Viewer<T>::builder()
                                   // Attach the full simulation system.
                                   .with_system(system)

                                   // Configure the viewer window.
                                   .with_title("Atlas DSMC Nitrogen Honeycomb Flow")
                                   .with_size(1440, 900)
                                   .with_background_color(Vector4<T>(1.f, 1.f, 1.f, 1.f))

                                   // Build the viewer.
                                   .build();

    // Frame the initial camera around the padded simulation bounds.
    viewer.camera().fit_bounds(domain_bounds.lower_corner, domain_bounds.upper_corner);

    // Render the live particle cloud.
    viewer.add_layer(
        vizkit::ParticleLayer<T>::builder()
            .with_system(system)
            .with_color(Vector4<T>(0.0f, 0.32f, 1.0f, 0.50f))
            .with_point_size(8.0f)
            .make_shared());

    // Render the imported honeycomb mesh as a semi-transparent reference layer.
    const auto honeycomb_layer = vizkit::TriangleMeshLayer<T>::builder()
                                     .with_unit(honeycomb_unit)
                                     .make_shared();
    honeycomb_layer->set_color(Vector4<T>(0.8f, 0.8f, 0.8f, 0.28f));
    honeycomb_layer->set_edge_color(Vector4<T>(0.f, 0.f, 0.f, 1.f));
    honeycomb_layer->set_line_width(1.5f);
    viewer.add_layer(honeycomb_layer);

    std::cout
        << "Nitrogen DSMC example: 300 K, bulk inflow along +y, circular source below the honeycomb, "
           "flow through a static imported triangle-mesh collider.\n";

    const int exit_code = viewer.run();
    observer->export_csv(std::filesystem::path(__FILE__).parent_path() / "observer_output");
    return exit_code;
#else
    // -------------------------------------------------------------------------
    // 7. Headless time integration
    // -------------------------------------------------------------------------
    // Run a fixed number of update steps when Vizkit is not enabled.

    for (int step = 0; step < 1000; ++step) {
        // Advance the simulation by one timestep.
        system->update();
    }

    // Print a compact completion message and the number of active particles left
    // in the fluid container after all update steps.
    std::cout
        << "Nitrogen DSMC example ran headlessly for " << 1000 << " steps.\n"
        << "Active particles: " << fluid->particle_count() << '\n';

    observer->export_csv(std::filesystem::path(__FILE__).parent_path() / "observer_output");
    return 0;
#endif
}
