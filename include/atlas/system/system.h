#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generator.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/sink/sink.h>
#include <atlas/solver/solver.h>
#include <atlas/source/source.h>
#include <atlas/universe/universe.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace atlas {

/**
 * @brief Top-level simulation driver that owns every engine subsystem and runs one step.
 *
 * A @c System aggregates the whole simulation: the particle @c Fluid, the spatial
 * @c Universe grid and its per-cell state, the emission @c Source / @c Generator pairs,
 * the physics @c Solver list, the @c Collider and @c Sink boundaries, the per-cell solver
 * @c Codec. @c update() advances the world by one @c _dt by running the fixed six-phase
 * pipeline (emit → search → allocate → solve → advect → remove) and incrementing the step
 * counter.
 *
 * Ownership: every subsystem is held by a host smart pointer or a device buffer that the
 * @c System owns outright. The type is therefore move-only (copying would duplicate GPU
 * buffers with host-only copy semantics); it is default-constructible so it can sit in a
 * @c host_unique_ptr before @c Builder::build() populates it.
 *
 * @note All work is orchestrated from the host; the individual pipeline phases launch
 *       device kernels internally. Construct instances through @ref builder().
 * @see System::Builder for the validated construction path.
 */
class System final {
public:
    /** @brief Fluent builder that accumulates subsystems and validates before construction. */
    class Builder;

public:
    /** @brief Constructs an empty system with no subsystems and the default timestep. */
    System() = default;

    /**
     * @brief Constructs a fully populated system by taking ownership of every subsystem.
     *
     * Moves the host pointers and copies the collider/sink buffers into device buffers,
     * then derives dependent state: if a universe is present it builds the spatial-hashing
     * searcher over it, then initializes the required universe state columns via
     * @ref initialize_states().
     *
     * @param fluid      Particle store; ownership transferred. Must be non-null in practice.
     * @param universe   Spatial grid and per-cell state; ownership transferred.
     * @param solvers    Physics solvers, run in order each step; ownership transferred.
     * @param sources    Emission boundaries, index-paired with @p generators; transferred.
     * @param generators Velocity/species generators, one per source; transferred.
     * @param colliders  Collision boundaries; copied host→device into @c _colliders.
     * @param sinks      Removal boundaries; copied host→device into @c _sinks.
     * @param codec      Per-cell solver selector; ownership transferred. May be null.
     * @param dt         Fixed timestep in seconds advanced by each @ref update().
     */
    ATLAS_HOST
    System(FluidHostPtr fluid,
           UniverseHostPtr universe,
           HostBuffer<SolverHostPtr> solvers,
           HostBuffer<SourceHostPtr> sources,
           HostBuffer<GeneratorHostPtr> generators,
           const HostBuffer<Collider>& colliders,
           const HostBuffer<Sink>& sinks,
           CodecHostPtr codec,
           float dt);

    /** @brief Non-copyable: the system owns unique device buffers and host pointers. */
    System(const System&) = delete;

    /** @brief Move-constructs, transferring ownership of every subsystem. */
    System(System&&) noexcept = default;

    /** @brief Destroys the system and every subsystem it owns. */
    ~System() = default;

    /** @brief Non-copyable: see the deleted copy constructor. */
    System&
    operator=(const System&)
        = delete;

    /** @brief Move-assigns, transferring ownership of every subsystem. */
    System&
    operator=(System&&) noexcept = default;

    /**
     * @brief Returns a fresh, empty builder for assembling a system.
     * @return A default-constructed @ref Builder.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Ensures every universe state column the configured solvers need exists.
     *
     * No-op when there is no universe. Always provisions the allocated-solver column, then
     * for each solver adds the columns that solver requires (e.g. the DSMC counters). Called
     * once during construction; safe to call again after the cell count changes because it
     * resizes existing columns rather than reallocating blindly.
     */
    ATLAS_HOST void
    initialize_states();

    /**
     * @brief Advances the whole simulation by one timestep.
     *
     * Runs the six pipeline phases in order (@ref emit, @ref search, @ref allocate,
     * @ref solve, @ref advect, @ref remove), then increments @c _step.
     */
    ATLAS_HOST void
    update();

    /**
     * @brief Pipeline phase 1: spawn new particles from every source.
     *
     * For each source, in order, appends spawned positions past the current live count
     * (stopping when the fluid buffer is full), fills the new slots' velocities and species
     * via the paired generator, and finally advances each source boundary by @c _dt. Updates
     * the fluid's live particle count.
     */
    ATLAS_HOST void
    emit();

    /**
     * @brief Pipeline phase 2: bin live particles into universe cells.
     *
     * No-op when there is no searcher. Delegates to the spatial-hashing searcher to classify
     * each particle into its cell and populate the per-cell particle-count state.
     */
    ATLAS_HOST void
    search();

    /**
     * @brief Pipeline phase 3: choose a solver per cell.
     *
     * No-op when there is no codec. Delegates to the codec, which reads per-cell temperature
     * and particle count and writes the per-cell allocated-solver index that @ref solve()
     * later honours.
     */
    ATLAS_HOST void
    allocate();

    /**
     * @brief Pipeline phase 4: run each physics solver over the fluid.
     *
     * No-op unless a fluid, universe, searcher, and at least one solver are all present.
     * Gathers the searcher view once, then runs every solver in order, passing the solver's
     * own index so a cell can select which solver's work applies to it.
     */
    ATLAS_HOST void
    solve();

    /**
     * @brief Pipeline phase 5: move particles and resolve collisions, then advance colliders.
     *
     * No-op unless both position and velocity states exist. For each particle it sweeps the
     * timestep displacement against the colliders using a broad phase (swept segment vs. each
     * collider's world AABB) followed by a narrow phase on the survivors, resolves the nearest
     * hit if any, otherwise integrates the position straight ahead. Finally advances every
     * moving collider by @c _dt.
     */
    ATLAS_HOST void
    advect();

    /**
     * @brief Pipeline phase 6: despawn particles at sinks and compact the fluid.
     *
     * No-op unless there are sinks, live particles, and both position and velocity states.
     * Marks survivors, compacts the fluid columns down to the survivors, then advances every
     * moving sink by @c _dt.
     */
    ATLAS_HOST void
    remove();

    /**
     * @brief Returns the fixed timestep advanced by each @ref update(), in seconds.
     * @return The timestep @c _dt.
     */
    ATLAS_NODISCARD ATLAS_HOST float
    dt() const noexcept {
        return _dt;
    }

    /**
     * @brief Returns the owned fluid handle.
     * @return Const reference to the fluid host pointer; may be null on an empty system.
     */
    ATLAS_NODISCARD ATLAS_HOST const FluidHostPtr&
    fluid() const noexcept {
        return _fluid;
    }

    /**
     * @brief Returns the owned universe handle.
     * @return Const reference to the universe host pointer; may be null on an empty system.
     */
    ATLAS_NODISCARD ATLAS_HOST const UniverseHostPtr&
    universe() const noexcept {
        return _universe;
    }

    /**
     * @brief Returns the spatial-hashing searcher built over the universe.
     * @return Const reference to the searcher host pointer; null until a universe is present.
     */
    ATLAS_NODISCARD ATLAS_HOST const SpatialHashingSearcherHostPtr&
    searcher() const noexcept {
        return _searcher;
    }

    /**
     * @brief Returns the ordered list of physics solvers.
     * @return Const reference to the solver buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST const HostBuffer<SolverHostPtr>&
    solvers() const noexcept {
        return _solvers;
    }

    /**
     * @brief Returns the per-cell solver selector.
     * @return Const reference to the codec host pointer; may be null.
     */
    ATLAS_NODISCARD ATLAS_HOST const CodecHostPtr&
    codec() const noexcept {
        return _codec;
    }

    /**
     * @brief Writes a binary snapshot of the fluid and universe for the current step.
     *
     * Creates a per-step subdirectory (named by @ref snapshot_directory_name) under
     * @p directory and serializes @c fluid.bin and @c universe.bin into it.
     *
     * @param directory Root output directory; the step subdirectory is created if missing.
     * @throws std::runtime_error propagated from the serializers on an I/O failure.
     */
    ATLAS_HOST void
    save(const std::filesystem::path& directory) const;

    /**
     * @brief Builds the conventional snapshot subdirectory name for a step.
     * @param step Zero-based step index.
     * @return The string @c "time_step_<step>".
     */
    ATLAS_NODISCARD ATLAS_HOST static std::string
    snapshot_directory_name(std::size_t step);

    /**
     * @brief Returns the number of completed steps.
     * @return The step counter @c _step, incremented once per @ref update().
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    step() const noexcept {
        return _step;
    }

    /**
     * @brief Returns the number of emission sources.
     * @return @c _sources.size().
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    source_count() const noexcept {
        return _sources.size();
    }

    /**
     * @brief Returns the number of physics solvers.
     * @return @c _solvers.size().
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    solver_count() const noexcept {
        return _solvers.size();
    }

    /**
     * @brief Returns the number of collision boundaries.
     * @return @c _colliders.size().
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    collider_count() const noexcept {
        return _colliders.size();
    }

    /**
     * @brief Returns the number of removal boundaries.
     * @return @c _sinks.size().
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    sink_count() const noexcept {
        return _sinks.size();
    }

    /**
     * @brief Returns the particles emitted by each source during the latest emit phase.
     * @return Host buffer indexed in the same order as the configured sources.
     */
    ATLAS_NODISCARD ATLAS_HOST const HostBuffer<int>&
    source_spawned_last_step() const noexcept {
        return _source_spawned_last_step;
    }

    /**
     * @brief Copies the particles removed by each sink during the latest remove phase.
     * @return Host vector indexed in the same order as the configured sinks.
     * @note CUDA builds perform one device-to-host transfer for this small counter buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::vector<int>
    sink_removed_last_step() const;

    /** @brief Returns current source poses in source insertion order. */
    ATLAS_NODISCARD ATLAS_HOST std::vector<Sync>
    source_syncs() const;

    /** @brief Copies current collider poses to the host in collider insertion order. */
    ATLAS_NODISCARD ATLAS_HOST std::vector<Sync>
    collider_syncs() const;

    /** @brief Copies current sink poses to the host in sink insertion order. */
    ATLAS_NODISCARD ATLAS_HOST std::vector<Sync>
    sink_syncs() const;

    /**
     * @brief Flags each live particle as surviving (1) or despawned (0) against the sinks.
     *
     * For every particle it tests each sink's despawn predicate; the first sink that claims
     * the particle clears its active flag. The flags become the scan input that
     * @c Fluid::compact() consumes in @ref remove().
     *
     * @param particle_count Number of leading live particles to test.
     * @note Public only because its body launches an extended @c __host__ __device__ lambda,
     *       which nvcc forbids inside a private or protected member function; it is an
     *       internal step of @ref remove() and not meant to be called directly.
     */
    ATLAS_HOST void
    mark_survivors(int particle_count);

private:
    /**
     * @brief Provisions the universe state columns the DSMC solver requires.
     *
     * Ensures the number-particle, max-relative-speed, max-sigma-g, and collision-count
     * per-cell columns exist and are sized to the cell count. Invoked by
     * @ref initialize_states() for each DSMC solver.
     */
    ATLAS_HOST void
    initialize_dsmc_states();

private:
    float _dt = 0.01f; ///< Fixed timestep advanced per @ref update(), in seconds.

    FluidHostPtr _fluid {}; ///< Owned particle store; the simulation subject.

    UniverseHostPtr _universe {}; ///< Owned spatial grid and per-cell state.

    SpatialHashingSearcherHostPtr _searcher {}; ///< Cell binning; built over @c _universe.

    std::size_t _step = 0; ///< Number of completed @ref update() calls.

    HostBuffer<SolverHostPtr> _solvers; ///< Physics solvers, run in order each step.

    CodecHostPtr _codec {}; ///< Per-cell solver selector; may be null.

    HostBuffer<SourceHostPtr> _sources; ///< Emission boundaries, index-paired with @c _generators.

    HostBuffer<GeneratorHostPtr> _generators; ///< Velocity/species generators, one per source.

    HostBuffer<int> _source_spawned_last_step; ///< Per-source counts from the latest emit phase.

    DeviceBuffer<Collider> _colliders; ///< Collision boundaries, device-resident for @ref advect().

    DeviceBuffer<Sink> _sinks; ///< Removal boundaries, device-resident for @ref remove().

    DeviceBuffer<int> _sink_removed_last_step; ///< Per-sink counts from the latest remove phase.
};

/**
 * @brief Fluent builder that accumulates subsystems and validates before constructing a System.
 *
 * Each @c with_* setter records one subsystem and returns @c *this for chaining. @ref build()
 * runs @ref validate() (which throws on a missing or inconsistent configuration) and then
 * moves the accumulated subsystems into a @c System. Setters that append (solvers, emitters,
 * colliders, sinks) may be called repeatedly; @c with_emitter keeps the source and generator
 * lists index-aligned.
 */
class System::Builder final {
public:
    /** @brief Constructs an empty builder with the default timestep and no subsystems. */
    Builder() = default;

    /**
     * @brief Sets the fluid (required).
     * @param fluid Particle store; ownership transferred.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    /**
     * @brief Sets the universe (required).
     * @param universe Spatial grid and per-cell state; ownership transferred.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    /**
     * @brief Appends a physics solver; solvers run in insertion order each step.
     * @param solver Solver to add; ownership transferred. Must be non-null at build time.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_solver(SolverHostPtr solver);

    /**
     * @brief Appends an emission source paired with its velocity/species generator.
     *
     * The two lists are kept index-aligned so @c emit() can pair source @c i with
     * generator @c i.
     *
     * @param source    Emission boundary; ownership transferred. Non-null at build time.
     * @param generator Generator that fills spawned slots; ownership transferred. Non-null
     *                  at build time.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_emitter(SourceHostPtr source, GeneratorHostPtr generator);

    /**
     * @brief Appends a collision boundary.
     * @param collider Collider, copied into the builder's host buffer.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_collider(const Collider& collider);

    /**
     * @brief Appends a removal boundary.
     * @param sink Sink, copied into the builder's host buffer.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_sink(const Sink& sink);

    /**
     * @brief Sets the per-cell solver selector (optional).
     * @param codec Codec; ownership transferred. May be left null.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_codec(CodecHostPtr codec) noexcept;

    /**
     * @brief Sets the fixed timestep in seconds (must be positive).
     * @param dt Timestep; validated to be strictly greater than zero at build time.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_dt(float dt) noexcept;

    /**
     * @brief Validates the configuration and constructs the system by value.
     * @return The fully constructed @c System.
     * @throws std::runtime_error if a required field is missing or inconsistent
     *         (see @ref validate()).
     */
    ATLAS_NODISCARD ATLAS_HOST System
    build();

    /**
     * @brief Validates, builds, and wraps the system in a host unique pointer.
     * @return A @c host_unique_ptr owning the newly built system.
     * @throws std::runtime_error propagated from @ref build().
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_unique_ptr<System>
    make_host_unique();

private:
    /**
     * @brief Throws if the accumulated configuration is missing a required field or is
     *        inconsistent.
     *
     * Requires a fluid, a universe, a positive @c _dt, equal source/generator counts with no
     * null entries, and no null solver.
     *
     * @throws std::runtime_error describing the first failed requirement.
     */
    ATLAS_HOST void
    validate() const;

private:
    float _dt = 0.01f; ///< Timestep to install; must be positive.

    FluidHostPtr _fluid {}; ///< Fluid to install (required).

    UniverseHostPtr _universe {}; ///< Universe to install (required).

    HostBuffer<SolverHostPtr> _solvers; ///< Solvers to install, in insertion order.

    CodecHostPtr _codec {}; ///< Codec to install; may be null.

    HostBuffer<SourceHostPtr> _sources; ///< Sources to install, index-aligned with @c _generators.

    HostBuffer<GeneratorHostPtr> _generators; ///< Generators to install, one per source.

    HostBuffer<Collider> _colliders; ///< Colliders to install.

    HostBuffer<Sink> _sinks; ///< Sinks to install.
};

/** @brief Owning host handle to a @ref System. */
using SystemHostPtr = atlas::host_unique_ptr<System>;

}
