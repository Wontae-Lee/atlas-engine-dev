#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <filesystem>
#include <iostream>

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
    HostBuffer<MaterialProperties<T>> properties(1);

    // Store one generator. Its index is expected to match the material/species setup.
    HostBuffer<GeneratorHostPtr<T>> generators(1);

    // Configure the nitrogen molecule properties.
    properties[0] = MaterialProperties<T>::builder()
                        // Mark this species as a molecule rather than a wall, marker,
                        // or other possible material category.
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

                        // Add no macroscopic drift velocity; only thermal velocity is sampled.
                        .with_bulk_velocity(Vector3F(0.0f, 0.0f, 0.0f))

                        // Use a fixed seed so generated velocity samples are reproducible.
                        .with_seed(42u)

                        // Allocate the generator in host-managed shared ownership.
                        .make_host_shared();

    // Create the fluid particle storage and attach material/generator metadata.
    const auto fluid = Fluid<T>::builder()
                           // Reserve storage for up to 200,000 particles.
                           .with_buffer_size(200000)

                           // Attach the single-species material table.
                           .with_properties(properties)

                           // Attach the velocity generator table used by source injection.
                           .with_generators(generators)

                           // Attach the observer used by runtime metrics.
                           .with_observer(observer)

                           // Allocate the fluid object in host-managed shared ownership.
                           .make_host_shared();

    // -------------------------------------------------------------------------
    // 2. Geometric regions
    // -------------------------------------------------------------------------
    // This block creates the open cylinder collider, its axis-aligned bounding
    // box domain, and the circular source disk.

    // Build the open cylinder used as the inward-facing collider wall.
    const auto cylinder_geometry = geometry::Cylinder<T>::builder()
                                       // Center the cylinder at the origin.
                                       .with_center(Vector3F(0, 0, 0))

                                       // Set the cylinder radius.
                                       .with_radius(1.0f)

                                       // Set the cylinder length along its axis.
                                       .with_height(8.0f)

                                       // Keep only the curved side wall.
                                       .with_open(true)

                                       // Allocate the geometry in host-managed shared ownership.
                                       .make_host_shared();

    // Build the simulation domain from the cylinder's axis-aligned bounds.
    const auto bounds = cylinder_geometry->bound();
    const auto domain_geometry = geometry::Box<T>::builder()
                                     // Match the cylinder lower bound.
                                     .with_lower_corner(bounds.lower_corner)

                                     // Match the cylinder upper bound.
                                     .with_upper_corner(bounds.upper_corner)

                                     // Allocate the geometry in host-managed shared ownership.
                                     .make_host_shared();

    // Build the circular source near the +z side of the cylinder.
    const auto source_geometry = geometry::Circle<T>::builder()
                                     // Place the source disk near the positive z side.
                                     .with_center(Vector3F(0.0f, 0.0f, 3.6f))

                                     // Orient the source disk parallel to the xy-plane.
                                     .with_normal(Vector3F(0.0f, 0.0f, 1.0f))

                                     // Fill most of the cylinder cross-section.
                                     .with_radius(1.0f)

                                     // Allocate the geometry in host-managed shared ownership.
                                     .make_host_shared();

    // -------------------------------------------------------------------------
    // 3. Core simulation containers
    // -------------------------------------------------------------------------
    // This block creates the universe, spatial searcher, DSMC solver, field
    // measurer, and orchestrator.

    // Create the Cartesian simulation universe from the cylinder bounds.
    const auto universe = Universe<T>::builder()
                              // Define the lower corner of the computational domain.
                              .with_lower_corner(bounds.lower_corner)

                              // Define the upper corner of the computational domain.
                              .with_upper_corner(bounds.upper_corner)

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

    const auto cylinder_unit = Unit<T>::builder()
                                   // Attach the open cylinder geometry.
                                   .with_geometry(cylinder_geometry)

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
                            .with_spacing(0.15f)

                            // Use the same thermal temperature as the velocity generator.
                            .with_temperature(300.0f)

                            // Allocate the source in host-managed shared ownership.
                            .make_host_shared();

    // Configure particle removal at the cylinder bounding box.
    const auto sink = fluid::Sink<T>::builder()
                          // Use the domain unit as the sink reference region.
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

    // Configure particle interaction with the inner cylinder wall.
    const auto collider = Collider<T>::builder()
                              // Attach the fluid whose particles will be tested against the collider.
                              .with_fluid(fluid)

                              // Use the cylinder unit as the wall collider.
                              .with_units(HostBuffer<Unit<T>> { *cylinder_unit })

                              // Define one surface-interaction model for the cylinder wall.
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

                              // Interpret the open cylinder as an inward-facing boundary.
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
                            .with_dt(2.5e-5f)

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
                                   .with_title("Atlas DSMC Nitrogen In-Cylinder Flow")
                                   .with_size(1440, 900)

                                   // Build the viewer.
                                   .build();

    // Fit the initial camera to the cylinder bounding box.
    viewer.camera().fit_bounds(bounds.lower_corner, bounds.upper_corner);

    // Render the live particle cloud.
    viewer.add_layer(
        vizkit::ParticleLayer<T>::builder()
            .with_system(system)
            .with_color(Vector4<T>(0.10f, 0.74f, 0.92f, 0.80f))
            .make_shared());

    // Render the domain bounding box as a wireframe reference.
    viewer.add_layer(
        vizkit::BoxLayer<T>::builder()
            .with_unit(domain_unit)
            .make_shared());

    // Render the cylinder as a semi-transparent wireframe reference.
    const auto cylinder_layer = vizkit::CylinderLayer<T>::builder()
                                    .with_unit(cylinder_unit)
                                    .with_slices(72)
                                    .make_shared();
    cylinder_layer->set_color(Vector4<T>(0.92f, 0.96f, 0.98f, 0.28f));
    viewer.add_layer(cylinder_layer);

    std::cout
        << "Nitrogen DSMC example: 300 K, zero bulk drift, circular source near +z, flow inside a finite cylinder.\n";

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
