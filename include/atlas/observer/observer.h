#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe.h>

#include <cstddef>
#include <filesystem>
#include <utility>

namespace atlas {

/**
 * @brief Periodic diagnostics writer: dumps the simulation state to CSV files.
 *
 * An @c Observer is the engine's output sink. Every @c interval() steps it is asked to
 * @c observe() the current @c Fluid and @c Universe and it writes one snapshot of CSV
 * files under @c output_directory()/data — one row per particle for the fluid, one row
 * per cell for the universe, plus per-source spawn and per-sink despawn tallies. Between
 * snapshots it holds nothing but two small device-side counter buffers; it does not own,
 * mutate, or step the simulation, so a run with no @c Observer behaves identically minus
 * the file output.
 *
 * The type is move-only: it owns @c DeviceBuffer members (thrust::device_vector) whose
 * copy is host-only and semantically an alias trap, so copies are deleted and the object
 * is passed by move or through an @c ObserverHostPtr. It is constructed exclusively via
 * its @ref Builder; the default constructor exists only so a moved-from or freshly built
 * shell is valid.
 *
 * @note The spawn/despawn counters live on the device because they are incremented from
 *       device kernels (@c System::record_spawned and @c System::mark_survivors). The
 *       @c Observer merely sizes, resets, and serializes them; @c System does the
 *       atomic accumulation.
 */
class Observer final {
public:
    /** @brief Fluent construction helper for @c Observer; see @ref Observer::Builder. */
    class Builder;

public:
    /** @brief Constructs an empty observer with a zero interval and no counters. */
    Observer() = default;

    /** @brief Deleted: an observer owns device buffers and is therefore non-copyable. */
    Observer(const Observer&) = delete;

    /** @brief Move-constructs, transferring ownership of the counter buffers. */
    Observer(Observer&&) noexcept = default;

    /** @brief Trivial destructor; the owned @c DeviceBuffer members release themselves. */
    ~Observer() = default;

    /** @brief Deleted copy assignment; see the deleted copy constructor. */
    Observer&
    operator=(const Observer&)
        = delete;

    /** @brief Move-assigns, transferring ownership of the counter buffers. */
    Observer&
    operator=(Observer&&) noexcept = default;

    /**
     * @brief Returns a fresh @ref Builder for configuring an @c Observer.
     * @return A default-constructed builder with a zero interval and empty output path.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Writes a CSV snapshot of the simulation when @p step lands on the interval.
     *
     * Returns immediately (a no-op) unless @c interval() is non-zero and @p step is an
     * exact multiple of it, so most steps cost nothing. On a snapshot step it creates
     * @c output_directory()/data and writes up to four files suffixed with the step
     * number: @c fluid_<step>.csv (one row per live particle), @c universe_<step>.csv
     * (one row per cell), @c source_<step>.csv and @c sink_<step>.csv (spawn/despawn
     * tallies). A file is skipped when its subject is empty — no particles, no cells, or
     * an empty counter buffer — so the set of files present records what actually
     * existed. Each file's columns cover only the states the object currently carries;
     * an absent state simply contributes no column.
     *
     * This copies the relevant device buffers to the host to serialize them, so it is a
     * synchronizing, comparatively expensive operation — the interval gate is what keeps
     * it affordable.
     *
     * @param fluid The particle set to serialize; its live @c particle_count() rows.
     * @param universe The background grid to serialize; its @c cell_count() rows.
     * @param step The current global step index, matched against @c interval().
     * @throws std::runtime_error if an output file cannot be opened for writing.
     */
    ATLAS_HOST void
    observe(const Fluid& fluid, const Universe& universe, std::size_t step) const;

    /**
     * @brief (Re)allocates the spawn/despawn counters and records the species count.
     *
     * Sizes @c spawned() to @p source_count * @p species_count and @c despawned() to
     * @p sink_count * @p species_count, laid out row-major as [entity][species], and
     * zero-fills both. Call this once the source, sink, and species counts are known
     * (see @c System's constructor); it both allocates and clears, so no separate reset
     * is needed immediately after.
     *
     * @param source_count Number of emission sources (rows of the spawn counter).
     * @param sink_count Number of removal sinks (rows of the despawn counter).
     * @param species_count Number of particle species (columns of both counters); also
     *        cached and returned by @c species_count().
     */
    ATLAS_HOST void
    resize_counters(std::size_t source_count, std::size_t sink_count, std::size_t species_count);

    /**
     * @brief Zeroes the spawn and despawn counters in place without reallocating.
     *
     * Runs a device fill over both buffers, preserving their current sizes. Use it to
     * clear the running tallies (e.g. per output window) while leaving the layout that
     * @c resize_counters() established intact.
     */
    ATLAS_HOST void
    reset_counters();

    /**
     * @brief Returns the number of species, i.e. the column stride of both counters.
     * @return The species count last passed to @c resize_counters(), or 0 before then.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    species_count() const noexcept {
        return _species_count;
    }

    /**
     * @brief Mutable access to the per-source spawn counter (device buffer).
     *
     * Row-major [source][species] of length @c source_count * @c species_count. Exposed
     * mutably so @c System can raw-pointer into it and atomically increment cells from a
     * device kernel as particles are emitted.
     *
     * @return Reference to the spawn counter buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    spawned() noexcept {
        return _spawned;
    }

    /** @brief Const access to the per-source spawn counter. @see spawned() */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    spawned() const noexcept {
        return _spawned;
    }

    /**
     * @brief Mutable access to the per-sink despawn counter (device buffer).
     *
     * Row-major [sink][species] of length @c sink_count * @c species_count. Exposed
     * mutably so @c System can atomically increment cells from the removal kernel.
     *
     * @return Reference to the despawn counter buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    despawned() noexcept {
        return _despawned;
    }

    /** @brief Const access to the per-sink despawn counter. @see despawned() */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    despawned() const noexcept {
        return _despawned;
    }

    /**
     * @brief Returns the snapshot cadence in steps.
     * @return The step interval; 0 disables output entirely (@c observe() is a no-op).
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    interval() const noexcept {
        return _interval;
    }

    /**
     * @brief Returns the root directory under which the @c data subdirectory is written.
     * @return The configured output directory path (may be empty if never set).
     */
    ATLAS_NODISCARD ATLAS_HOST const std::filesystem::path&
    output_directory() const noexcept {
        return _output_directory;
    }

private:
    /** @brief Grants the builder write access to the private configuration fields. */
    friend class Builder;

    std::size_t _interval = 0; ///< Snapshot cadence in steps; 0 disables output.

    std::filesystem::path _output_directory; ///< Root dir; snapshots go in its @c data child.

    std::size_t _species_count = 0; ///< Column stride of both counters; species count.

    DeviceBuffer<int> _spawned; ///< Row-major [source][species] spawn tally (device).

    DeviceBuffer<int> _despawned; ///< Row-major [sink][species] despawn tally (device).
};

/**
 * @brief Fluent builder that configures and produces an @c Observer.
 *
 * Collects the two build-time settings — the snapshot interval and the output
 * directory — through chainable @c with_* setters, then materializes an @c Observer via
 * @c build() or an @c ObserverHostPtr via @c make_host_shared(). The counter buffers are
 * not configured here; they are sized later by @c Observer::resize_counters() once the
 * source, sink, and species counts are known. There is no @c validate() step because
 * both fields have valid defaults (a zero interval simply disables output).
 */
class Observer::Builder final {
public:
    /** @brief Constructs a builder with a zero interval and an empty output directory. */
    Builder() = default;

    /**
     * @brief Sets the snapshot cadence in steps.
     * @param interval Steps between snapshots; 0 leaves output disabled.
     * @return @c *this, for chaining.
     */
    ATLAS_HOST Builder&
    with_interval(std::size_t interval) noexcept;

    /**
     * @brief Sets the root output directory (snapshots land in its @c data child).
     * @param output_directory Destination path; taken by value and moved into the builder.
     * @return @c *this, for chaining.
     */
    ATLAS_HOST Builder&
    with_output_directory(std::filesystem::path output_directory);

    /**
     * @brief Builds a value @c Observer from the accumulated settings.
     * @return An @c Observer carrying the configured interval and directory, with empty
     *         (unsized) counters awaiting @c Observer::resize_counters().
     */
    ATLAS_NODISCARD ATLAS_HOST Observer
    build() const;

    /**
     * @brief Builds an @c Observer and wraps it in a host-side shared pointer.
     *
     * Equivalent to @c build() followed by @c make_host_shared, moving the built value
     * into the shared owner. This is the form @c System takes, since the observer must
     * outlive individual step calls and is shared through the system's wiring.
     *
     * @return A shared owner of a heap-allocated @c Observer.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Observer>
    make_host_shared() const;

private:
    std::size_t _interval = 0;               ///< Pending snapshot cadence copied into the built observer.
    std::filesystem::path _output_directory; ///< Pending output root copied into the observer.
};

/** @brief Host-side shared-ownership handle for an @c Observer (the form @c System holds). */
using ObserverHostPtr = host_shared_ptr<Observer>;

/** @brief Device-side shared handle alias for symmetry with other modules; unused here. */
using ObserverDevicePtr = device_shared_ptr<Observer>;

}