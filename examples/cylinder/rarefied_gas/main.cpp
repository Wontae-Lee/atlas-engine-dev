#include <atlas/atlas.h>

#include <iostream>

using namespace atlas;

int
main() {
    // Use single precision for this example to reduce memory traffic and improve
    // throughput in particle-heavy DSMC-style simulations.
    using T = float;

    // -------------------------------------------------------------------------
    // 1. Material and velocity-generation configuration
    // -------------------------------------------------------------------------
    // This block defines the physical species and the initial velocity sampler.
    // The material properties describe what a particle "is", while the generator
    // describes how its initial velocity is sampled during spawning.

    // Store one material species. The current example models a single molecular gas.
    HostBuffer<MaterialProperties<T>> properties(1);

    // Store one generator. Its index is expected to match the material/species setup.
    HostBuffer<GeneratorHostPtr<T>> generators(1);

    // Configure the gas molecule properties.
    properties[0] = MaterialProperties<T>::builder()
                        // Mark this species as a molecule rather than a wall, marker,
                        // or other possible material category.
                        .with_type(MaterialType::Molecule)

                        // Set the particle mass in kilograms.
                        // This value is used by kinetic models and collision routines.
                        .with_mass(4.651734e-26f)

                        // Set the molecular mass used by thermal velocity generation.
                        // Keeping this equal to mass is appropriate for this single-species
                        // molecule setup.
                        .with_molecular_mass(4.651734e-26f)

                        // Assign a species identifier. With only one species, zero is used.
                        .with_species_id(0)

                        // Set the reference collision diameter used by hard-sphere
                        // collision models.
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
                        .with_bulk_velocity(Vector3F(0, 0, 0))

                        // Use a fixed seed so generated velocity samples are reproducible.
                        .with_seed(42u)

                        // Allocate the generator in host-managed shared ownership.
                        .make_host_shared();

    // -------------------------------------------------------------------------
    // 2. Core simulation containers
    // -------------------------------------------------------------------------
    // This block creates the particle container, spatial domain, spatial searcher,
    // DSMC solver, field measurer, and orchestrator. These objects define the
    // simulation state and the main numerical update pipeline.

    // Create the fluid particle storage and attach material/generator metadata.
    const auto fluid = Fluid<T>::builder()
                           // Reserve storage for up to 200,000 particles.
                           // This controls capacity, not necessarily the initial
                           // active particle count.
                           .with_buffer_size(200000)

                           // Attach the single-species material table.
                           .with_properties(properties)

                           // Attach the velocity generator table used by source injection.
                           .with_generators(generators)

                           // Allocate the fluid object in host-managed shared ownership.
                           .make_host_shared();

    // Create the Cartesian simulation universe.
    const auto universe = Universe<T>::builder()
                              // Define the lower corner of the computational domain.
                              .with_lower_corner(Vector3F(-6.0f, -2.5f, -1.2f))

                              // Define the upper corner of the computational domain.
                              .with_upper_corner(Vector3F(6.0f, 2.5f, 1.2f))

                              // Use a uniform cell size for spatial hashing and cell-wise
                              // DSMC operations.
                              .with_cell_size(0.25f)

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
    const auto dsmc_solver = make_host_shared<DsmcSolver<T>>(
        universe,
        fluid,
        searcher,
        system::DsmcKernelType::hard_sphere);

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
    // 3. Shared transform state
    // -------------------------------------------------------------------------
    // Geometry objects are wrapped in units. A unit combines geometry with a
    // synchronization object that stores its rigid pose. Here all geometries use
    // the same identity transform.

    const auto sync = Sync<T>::builder()
                          // Place the geometry at the origin with identity rotation.
                          // Quaternion(1, 0, 0, 0) represents no rotation.
                          .with_rigid_pose(Vector3F(0, 0, 0), Quaternion<T>(1, 0, 0, 0))

                          // Allocate the sync object in host-managed shared ownership.
                          .make_host_shared();

    // -------------------------------------------------------------------------
    // 4. Geometric regions
    // -------------------------------------------------------------------------
    // This block defines three physical regions:
    //   - domain_geometry: the full simulation volume used by the sink.
    //   - source_geometry: a small injection volume near the left side.
    //   - cylinder_geometry: an internal solid obstacle used by the collider.

    // Full simulation box used as the outer control volume.
    const auto domain_geometry = geometry::Box<T>::builder()
                                     // Match the lower corner of the universe domain.
                                     .with_lower_corner(Vector3F(-6.0f, -2.5f, -1.2f))

                                     // Match the upper corner of the universe domain.
                                     .with_upper_corner(Vector3F(6.0f, 2.5f, 1.2f))

                                     // Allocate the geometry in host-managed shared ownership.
                                     .make_host_shared();

    // Local source box where new particles are injected.
    const auto source_geometry = geometry::Box<T>::builder()
                                     // Place the source near the left side of the domain.
                                     .with_lower_corner(Vector3F(-5.1f, -0.9f, -0.18f))

                                     // Give the source a finite cross-sectional area.
                                     .with_upper_corner(Vector3F(-4.0f, 0.9f, 0.18f))

                                     // Allocate the geometry in host-managed shared ownership.
                                     .make_host_shared();

    // Cylindrical obstacle placed near the center of the domain.
    const auto cylinder_geometry = geometry::Cylinder<T>::builder()
                                       // Center the cylinder at the origin.
                                       .with_center(Vector3F(0, 0, 0))

                                       // Set the cylinder radius in the x-y plane.
                                       .with_radius(0.9f)

                                       // Set the cylinder height along the cylinder axis.
                                       .with_height(2.2f)

                                       // Allocate the geometry in host-managed shared ownership.
                                       .make_host_shared();

    // -------------------------------------------------------------------------
    // 5. Units: geometry plus transform
    // -------------------------------------------------------------------------
    // Units bind a geometry object to a pose/sync object. This allows sources,
    // sinks, and colliders to consume a common geometric interface while still
    // supporting transformed objects.

    const auto domain_unit = Unit<T>::builder()
                                 // Attach the full-domain geometry.
                                 .with_geometry(domain_geometry)

                                 // Attach the shared identity transform.
                                 .with_sync(sync)

                                 // Allocate the unit in host-managed shared ownership.
                                 .make_host_shared();

    const auto source_unit = Unit<T>::builder()
                                 // Attach the source injection geometry.
                                 .with_geometry(source_geometry)

                                 // Attach the shared identity transform.
                                 .with_sync(sync)

                                 // Allocate the unit in host-managed shared ownership.
                                 .make_host_shared();

    const auto cylinder_unit = Unit<T>::builder()
                                   // Attach the cylindrical obstacle geometry.
                                   .with_geometry(cylinder_geometry)

                                   // Attach the shared identity transform.
                                   .with_sync(sync)

                                   // Allocate the unit in host-managed shared ownership.
                                   .make_host_shared();

    // -------------------------------------------------------------------------
    // 6. Boundary and interaction systems
    // -------------------------------------------------------------------------
    // This block defines how particles enter, leave, and interact with geometry:
    //   - source: injects particles inside the source volume.
    //   - sink: removes particles outside/inside a selected region depending on flip.
    //   - collider: handles particle-surface interaction with the cylinder.

    // Configure particle injection.
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
                            .with_spacing(0.18f)

                            // Use the same thermal temperature as the velocity generator.
                            .with_temperature(300.0f)

                            // Allocate the source in host-managed shared ownership.
                            .make_host_shared();

    // Configure particle removal at the domain boundary.
    const auto sink = fluid::Sink<T>::builder()
                          // Use the full domain unit as the sink reference region.
                          .with_units(HostBuffer<Unit<T>> { *domain_unit })

                          // Attach the particle container from which particles are removed.
                          .with_fluid(fluid)

                          // Evaluate despawning using the volume of the domain geometry.
                          .with_despawn_types(HostBuffer<fluid::DespawnType> {
                              fluid::DespawnType::Volume,
                          })

                          // Use a volume despawn operator matching the selected despawn type.
                          .with_despawn_operator(fluid::DespawnOperator<T>(fluid::DespawnType::Volume))

                          // Flip the volume test so particles outside the domain are removed
                          // instead of particles inside the domain.
                          .with_flip(true)

                          // Allocate the sink in host-managed shared ownership.
                          .make_host_shared();

    // Configure particle interaction with the cylindrical obstacle.
    const auto collider = Collider<T>::builder()
                              // Attach the fluid whose particles will be tested against the collider.
                              .with_fluid(fluid)

                              // Use the cylinder unit as the solid obstacle.
                              .with_units(HostBuffer<Unit<T>> { *cylinder_unit })

                              // Define one surface-interaction model for the cylinder.
                              .with_surface_interactions(
                                  HostBuffer<system::IsothermalSurfaceInteraction<T>> {
                                      system::IsothermalSurfaceInteraction<T>::builder()
                                          // Use cosine-weighted diffuse reflection, which is a
                                          // common model for thermally accommodated gas-wall scattering.
                                          .with_diffuse_sampling(system::DiffuseSampling::CosineWeighted)

                                          // Keep the outgoing speed equal to the incoming speed.
                                          // A value below one would damp the reflected velocity.
                                          .with_restitution(1.0f)

                                          // Use full momentum accommodation.
                                          // With 1.0, the reflection branch is fully diffuse.
                                          .with_momentum_acc(1.0f)

                                          // Store the wall temperature used by the interaction model.
                                          .with_temperature(300.0f)

                                          // Finalize the surface-interaction object.
                                          .build(),
                                  })

                              // Allocate the collider in host-managed shared ownership.
                              .make_host_shared();

    // -------------------------------------------------------------------------
    // 7. Full system assembly
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

    // -------------------------------------------------------------------------
    // 8. Time integration
    // -------------------------------------------------------------------------
    // Advance the full coupled system. Each update typically performs particle
    // injection, boundary removal, collider interaction, spatial search refresh,
    // collision solving, and measurement depending on the configured system order.

    for (int step = 0; step < 1000; ++step) {
        // Advance the simulation by one timestep.
        system->update();
    }

    // -------------------------------------------------------------------------
    // 9. Final diagnostics
    // -------------------------------------------------------------------------
    // Print a compact completion message and the number of active particles left
    // in the fluid container after all update steps.

    std::cout << "Cylinder rarefied gas completed " << 1000 << " steps.\n"
              << "Active particles: " << fluid->particle_count() << '\n';

    return 0;
}
