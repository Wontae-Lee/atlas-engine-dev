#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>

#include <any>
#include <cstddef>
#include <utility>

/**
 * @file fluid_state.h
 * @brief The per-particle attribute buffers (position, velocity,
 *        species, active flag, temperature, internal energy) that make
 *        up a `Fluid`'s structure-of-arrays particle representation,
 *        plus the shared compaction/reset machinery every concrete
 *        state uses.
 *
 * @details
 * ### Operating principle
 * Each concrete `*State` class (`FluidPositionState`,
 * `FluidVelocityState`, ...) owns exactly one `DeviceBuffer<T>` and
 * implements the small `FluidState` interface (`size`/`compact`/
 * `reset`) so `Fluid` can hold an arbitrary, type-heterogeneous set of
 * them in one `FluidStateStore` (a `TypeStore<FluidState>`, see
 * `container/type_store.h`) and operate on all of them uniformly (e.g.
 * `Sink::compact_fluid_particles` iterates every registered state and
 * calls `compact()` on each, without knowing their concrete types).
 * Not every `Fluid` needs every state — a purely translational DSMC gas
 * has no `FluidInternalEnergyState`; systems check `has_state<T>()`
 * before touching a state that may not exist (as `DsmcSolver::solve`
 * and `ColliderCollisionKernel` do for internal energy).
 *
 * `compact_buffer`/`reset_buffer` are the shared implementations every
 * concrete state's `compact()`/`reset()` delegate to (via CRTP-free
 * composition — each state just calls
 * `compact_buffer(_position, compact_indices, kept)` etc. with its own
 * buffer):
 *   - `reset_buffer`: parallel-fills the buffer with value-initialized
 *     elements (`T{}`) — used to clear stale slots past the active
 *     particle count after compaction (see `Sink::compact_fluid_particles`).
 *   - `compact_buffer`: gathers `buffer[compact_indices[i]] -> scratch[i]`
 *     for `i` in `[0, kept)`, then copies the scratch buffer back over
 *     the original — the same gather-then-copy-back shape for every
 *     state, driven by one shared `compact_indices` permutation computed
 *     once by the caller (`Sink`'s stream compaction). The scratch
 *     buffer (`_compacted`, a type-erased `std::any` holding a
 *     `DeviceBuffer<value_type>`) is reused across calls rather than
 *     reallocated every compaction, at the cost of one `std::any_cast`
 *     per call — type erasure is needed here because the base class
 *     itself is not templated on `value_type` (each derived state's
 *     value type differs), so it cannot declare a strongly-typed member
 *     for the scratch buffer directly.
 */

namespace atlas {

/**
 * @brief One DSMC particle's internal (non-translational) energy modes;
 *        see `maxwellian_surface_interaction.h`'s Background and
 *        `dsmc_energy_exchange_solver.h` for how these are sampled and
 *        exchanged (Larsen-Borgnakke model).
 */
struct FluidInternalEnergy final {

    /** Translational kinetic energy remaining after internal-mode
     *  exchange (see `DsmcEnergyExchangeSolver::exchange_internal_energy`). */
    float translational {};

    /** Rotational mode energy. */
    float rotational {};

    /** Vibrational mode energy. */
    float vibrational {};
};

/**
 * @brief Base interface every per-particle attribute buffer implements,
 *        so `Fluid`'s heterogeneous `FluidStateStore` can operate on all
 *        registered states uniformly. See this file's top-of-file
 *        documentation for the compaction/reset machinery this provides
 *        to derived classes.
 */
class FluidState {
public:
    FluidState() = default;

    FluidState(const FluidState&) = delete;

    FluidState(FluidState&&) noexcept = default;

    virtual ~FluidState() = default;

    FluidState&
    operator=(const FluidState&)
        = delete;

    FluidState&
    operator=(FluidState&&) noexcept = default;

    /** @brief Current buffer capacity (not necessarily equal to
     *  `Fluid::particle_count()` — states are sized to the fluid's
     *  buffer capacity, not the live particle count). */
    ATLAS_HOST ATLAS_NODISCARD virtual std::size_t
    size() const noexcept = 0;

    /** @brief Gathers this state's buffer through `compact_indices`,
     *  keeping only the first `kept` entries; see this file's
     *  top-of-file documentation for the shared `compact_buffer` this
     *  delegates to. */
    ATLAS_HOST virtual void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept)
        = 0;

    /** @brief Value-initializes every element of this state's buffer;
     *  see this file's top-of-file documentation for the shared
     *  `reset_buffer` this delegates to. */
    ATLAS_HOST virtual void
    reset()
        = 0;

public:
    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    compact_buffer(Buffer& buffer,
                   const DeviceBuffer<std::size_t>& compact_indices,
                   const std::size_t kept) {

        if (kept == 0) {
            return;
        }

        using value_type = typename Buffer::value_type;

        if (!_compacted.has_value() || _compacted.type() != typeid(DeviceBuffer<value_type>)) {
            _compacted.emplace<DeviceBuffer<value_type>>();
        }

        auto& compacted = std::any_cast<DeviceBuffer<value_type>&>(_compacted);

        compacted.resize(kept);

        auto* dst          = atlas::raw_pointer_cast(compacted.data());
        auto* src          = atlas::raw_pointer_cast(buffer.data());
        const auto* source = atlas::raw_pointer_cast(compact_indices.data());

        atlas::parallel_for<ExecutionPolicy::device>(
            std::size_t { 0 },
            kept,
            [=] ATLAS_ALL_DEVICE(const std::size_t i) {
                dst[i] = src[source[i]];
            });

        atlas::parallel_for<ExecutionPolicy::device>(
            std::size_t { 0 },
            kept,
            [=] ATLAS_ALL_DEVICE(const std::size_t i) {
                src[i] = dst[i];
            });
    }

    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_buffer(Buffer& buffer) {
        using value_type = typename Buffer::value_type;

        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), value_type {});
    }

public:
    std::any _compacted;
};

/** @brief Per-particle world-space position buffer. Every concrete
 *  `*State` class in this file follows this same shape: an
 *  `explicit(buffer_size)` constructor, an `explicit(DeviceBuffer<T>)`
 *  adopting constructor, `data()` accessors, and `compact()`/`reset()`
 *  delegating to `FluidState::compact_buffer`/`reset_buffer`. */
class FluidPositionState final : public FluidState {
public:
    FluidPositionState() = default;

    ATLAS_HOST explicit FluidPositionState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidPositionState(DeviceBuffer<Vector3> position) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3> _position;
};

/** @brief Per-particle world-space velocity buffer. */
class FluidVelocityState final : public FluidState {
public:
    FluidVelocityState() = default;

    ATLAS_HOST explicit FluidVelocityState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidVelocityState(DeviceBuffer<Vector3> velocity) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3> _velocity;
};

/** @brief Per-particle species index buffer (indexes into
 *  `Fluid::particle_properties()`/`Fluid::generators()`). */
class FluidSpeciesState final : public FluidState {
public:
    FluidSpeciesState() = default;

    ATLAS_HOST explicit FluidSpeciesState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidSpeciesState(DeviceBuffer<std::size_t> species) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<std::size_t>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<std::size_t>&
    data() const noexcept;

private:
    DeviceBuffer<std::size_t> _species;
};

/** @brief Per-particle active flag buffer (`1` = live, `0` = pending
 *  removal); see `Sink::despawn_particles`/`compact_fluid_particles`. */
class FluidActiveState final : public FluidState {
public:
    FluidActiveState() = default;

    ATLAS_HOST explicit FluidActiveState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidActiveState(DeviceBuffer<int> active) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<int>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<int>&
    data() const noexcept;

private:
    DeviceBuffer<int> _active;
};

/** @brief Per-particle diagnostic temperature buffer; written by
 *  `BoltzmannMeasurer::assign_particle_temperature`, optional (only
 *  present when a measurer needs per-particle temperature output). */
class FluidTemperatureState final : public FluidState {
public:
    FluidTemperatureState() = default;

    ATLAS_HOST explicit FluidTemperatureState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidTemperatureState(DeviceBuffer<float> temperature) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<float>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _temperature;
};

/** @brief Per-particle `FluidInternalEnergy` buffer; optional (only
 *  present for gases modeling rotational/vibrational energy exchange,
 *  see `DsmcEnergyExchangeSolver`). */
class FluidInternalEnergyState final : public FluidState {
public:
    FluidInternalEnergyState() = default;

    ATLAS_HOST explicit FluidInternalEnergyState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidInternalEnergyState(DeviceBuffer<FluidInternalEnergy> internal_energy) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<FluidInternalEnergy>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<FluidInternalEnergy>&
    data() const noexcept;

private:
    DeviceBuffer<FluidInternalEnergy> _internal_energy;
};

}
