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

namespace atlas {

/**
 * @brief Abstract base for one device-resident per-particle attribute buffer.
 *
 * A @c Fluid keeps its particle attributes as a set of separate structure-of-arrays
 * columns (position, velocity, species, the energy modes), one @c FluidState leaf per
 * column. Storing each attribute behind this common base lets @c Fluid hold them
 * type-erased in a @c TypeStore and drive them uniformly: on a compaction pass it walks
 * every registered state and asks each to gather itself by the same index list, so a new
 * attribute can be added without touching the compaction loop.
 *
 * Each leaf owns exactly one @c DeviceBuffer, which is why the type is move-only
 * (@c DeviceBuffer's copy is host-only) and held through @c std::unique_ptr in the store.
 * All operations are host-side orchestration that launch device kernels internally.
 *
 * @note The buffers are sized to the fluid's @em capacity, not its live particle count;
 *       only the first @c particle_count entries are meaningful at any moment.
 */
class FluidState {
public:
    /** @brief Constructs an empty state; a derived class allocates its buffer. */
    FluidState() = default;

    /** @brief Deleted: a state owns a device buffer and is therefore non-copyable. */
    FluidState(const FluidState&) = delete;

    /** @brief Move-constructs, transferring ownership of the device buffer. */
    FluidState(FluidState&&) noexcept = default;

    /** @brief Virtual so leaves are destroyed correctly through a @c FluidState pointer. */
    virtual ~FluidState() = default;

    /** @brief Deleted: a state owns a device buffer and is therefore non-copyable. */
    FluidState&
    operator=(const FluidState&)
        = delete;

    /** @brief Move-assigns, transferring ownership of the device buffer. */
    FluidState&
    operator=(FluidState&&) noexcept = default;

    /**
     * @brief Returns the capacity of the underlying buffer, in elements.
     *
     * This is the allocated length (the fluid's buffer size), not the live particle
     * count; the caller tracks how many leading entries are alive.
     *
     * @return Number of elements the buffer can hold.
     */
    ATLAS_NODISCARD ATLAS_HOST virtual std::size_t
    size() const noexcept = 0;

    /**
     * @brief Gathers the surviving entries to the front of the buffer, in place.
     *
     * Given the source indices of the @p kept survivors, rewrites the buffer so that
     * element @c i becomes the old element at @c compact_indices[i], for @c i in
     * @c [0, kept). Every state in a fluid is compacted with the same index list so the
     * columns stay row-aligned. Runs on the device.
     *
     * @param compact_indices Device buffer of survivor source indices, length @p kept.
     * @param kept            Number of survivors; a value of 0 is a no-op.
     */
    ATLAS_HOST virtual void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept)
        = 0;

    /**
     * @brief Zero-initialises the entire buffer on the device.
     *
     * Overwrites every element with a value-initialised @c value_type, clearing any
     * previous particle data. Runs on the device.
     */
    ATLAS_HOST virtual void
    reset()
        = 0;

public:
    /**
     * @brief Compacts one owned buffer by gathering survivors to its front.
     *
     * Shared implementation the leaves call from their @c compact() override. It gathers
     * out-of-place into a scratch buffer and then copies the result back into @p buffer,
     * because gathering in place would let one survivor overwrite another that has not
     * been read yet. The scratch buffer (@c _compacted) is a @c std::any reused across
     * calls and re-created only when the element type changes, so repeated compactions of
     * the same-typed buffer avoid reallocation.
     *
     * @tparam Buffer          A @c DeviceBuffer specialisation; its @c value_type selects
     *                         the scratch type.
     * @param buffer           The device buffer to compact in place.
     * @param compact_indices  Device buffer of survivor source indices, length @p kept.
     * @param kept             Number of survivors; 0 returns immediately without touching
     *                         @p buffer.
     */
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

    /**
     * @brief Fills one owned buffer with value-initialised elements on the device.
     *
     * Shared implementation the leaves call from their @c reset() override.
     *
     * @tparam Buffer A @c DeviceBuffer specialisation.
     * @param buffer  The device buffer to clear over its full length.
     */
    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_buffer(Buffer& buffer) {
        using value_type = typename Buffer::value_type;

        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), value_type {});
    }

public:
    std::any _compacted; ///< Reused scratch buffer for out-of-place gather; holds a
                         ///< DeviceBuffer of the last-compacted element type.
};

/**
 * @brief Particle positions, one @c Float3 per particle, in simulation-space metres.
 *
 * The mandatory spatial column: every fluid allocates it. Motion integration, sink
 * despawn tests, and the spatial-hashing searcher all read it.
 */
class FluidPositionState final : public FluidState {
public:
    /** @brief Constructs an empty state with no allocation. */
    FluidPositionState() = default;

    /**
     * @brief Allocates a position buffer sized to the fluid's capacity.
     * @param buffer_size Number of particle slots to allocate.
     */
    ATLAS_HOST explicit FluidPositionState(std::size_t buffer_size);

    /**
     * @brief Adopts an existing device buffer as the position column.
     * @param position Device buffer to take ownership of by move.
     */
    ATLAS_HOST explicit FluidPositionState(DeviceBuffer<Float3> position) noexcept;

    /** @copydoc FluidState::size */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /** @copydoc FluidState::compact */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    /** @copydoc FluidState::reset */
    ATLAS_HOST void
    reset() override;

    /** @brief Returns a mutable reference to the owned position buffer. */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    /** @brief Returns a const reference to the owned position buffer. */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _position; ///< Per-particle position, in metres.
};

/**
 * @brief Particle velocities, one @c Float3 per particle, in metres per second.
 *
 * The mandatory kinematic column: every fluid allocates it. Read during motion
 * integration and by the DSMC collision kernel (through @c FluidDsmcView), which also
 * writes back post-collision velocities.
 */
class FluidVelocityState final : public FluidState {
public:
    /** @brief Constructs an empty state with no allocation. */
    FluidVelocityState() = default;

    /**
     * @brief Allocates a velocity buffer sized to the fluid's capacity.
     * @param buffer_size Number of particle slots to allocate.
     */
    ATLAS_HOST explicit FluidVelocityState(std::size_t buffer_size);

    /**
     * @brief Adopts an existing device buffer as the velocity column.
     * @param velocity Device buffer to take ownership of by move.
     */
    ATLAS_HOST explicit FluidVelocityState(DeviceBuffer<Float3> velocity) noexcept;

    /** @copydoc FluidState::size */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /** @copydoc FluidState::compact */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    /** @copydoc FluidState::reset */
    ATLAS_HOST void
    reset() override;

    /** @brief Returns a mutable reference to the owned velocity buffer. */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    /** @brief Returns a const reference to the owned velocity buffer. */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _velocity; ///< Per-particle velocity, in metres per second.
};

/**
 * @brief Per-particle species identifier, one @c std::size_t per particle.
 *
 * The mandatory material column: every fluid allocates it. The value indexes the fluid's
 * @c MaterialDictionary, selecting the particle's mass, collision cross-section, and
 * energy-mode properties. Read by the DSMC kernel and by the observer's per-species
 * despawn tallies.
 */
class FluidSpeciesState final : public FluidState {
public:
    /** @brief Constructs an empty state with no allocation. */
    FluidSpeciesState() = default;

    /**
     * @brief Allocates a species buffer sized to the fluid's capacity.
     * @param buffer_size Number of particle slots to allocate.
     */
    ATLAS_HOST explicit FluidSpeciesState(std::size_t buffer_size);

    /**
     * @brief Adopts an existing device buffer as the species column.
     * @param species Device buffer to take ownership of by move.
     */
    ATLAS_HOST explicit FluidSpeciesState(DeviceBuffer<std::size_t> species) noexcept;

    /** @copydoc FluidState::size */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /** @copydoc FluidState::compact */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    /** @copydoc FluidState::reset */
    ATLAS_HOST void
    reset() override;

    /** @brief Returns a mutable reference to the owned species buffer. */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<std::size_t>&
    data() noexcept;

    /** @brief Returns a const reference to the owned species buffer. */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<std::size_t>&
    data() const noexcept;

private:
    DeviceBuffer<std::size_t> _species; ///< Per-particle material-dictionary index.
};

/**
 * @brief Optional per-particle temperature column, one @c float per particle, in kelvin.
 *
 * Not allocated by default; register it on a fluid only when a solver or diagnostic needs
 * a per-particle temperature field.
 */
class FluidTemperatureState final : public FluidState {
public:
    /** @brief Constructs an empty state with no allocation. */
    FluidTemperatureState() = default;

    /**
     * @brief Allocates a temperature buffer sized to the fluid's capacity.
     * @param buffer_size Number of particle slots to allocate.
     */
    ATLAS_HOST explicit FluidTemperatureState(std::size_t buffer_size);

    /**
     * @brief Adopts an existing device buffer as the temperature column.
     * @param temperature Device buffer to take ownership of by move.
     */
    ATLAS_HOST explicit FluidTemperatureState(DeviceBuffer<float> temperature) noexcept;

    /** @copydoc FluidState::size */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /** @copydoc FluidState::compact */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    /** @copydoc FluidState::reset */
    ATLAS_HOST void
    reset() override;

    /** @brief Returns a mutable reference to the owned temperature buffer. */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /** @brief Returns a const reference to the owned temperature buffer. */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _temperature; ///< Per-particle temperature, in kelvin.
};

/**
 * @brief Optional per-particle translational-energy column, one @c float per particle, in
 *        joules.
 *
 * One of the three internal-energy modes DSMC can track. Not allocated by default;
 * register it only when the collision model exchanges translational energy explicitly.
 */
class FluidTranslationalEnergyState final : public FluidState {
public:
    /** @brief Constructs an empty state with no allocation. */
    FluidTranslationalEnergyState() = default;

    /**
     * @brief Allocates a translational-energy buffer sized to the fluid's capacity.
     * @param buffer_size Number of particle slots to allocate.
     */
    ATLAS_HOST explicit FluidTranslationalEnergyState(std::size_t buffer_size);

    /**
     * @brief Adopts an existing device buffer as the translational-energy column.
     * @param translational_energy Device buffer to take ownership of by move.
     */
    ATLAS_HOST explicit FluidTranslationalEnergyState(DeviceBuffer<float> translational_energy) noexcept;

    /** @copydoc FluidState::size */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /** @copydoc FluidState::compact */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    /** @copydoc FluidState::reset */
    ATLAS_HOST void
    reset() override;

    /** @brief Returns a mutable reference to the owned translational-energy buffer. */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /** @brief Returns a const reference to the owned translational-energy buffer. */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _translational_energy; ///< Per-particle translational energy, in joules.
};

/**
 * @brief Optional per-particle rotational-energy column, one @c float per particle, in
 *        joules.
 *
 * One of the three internal-energy modes DSMC can track. Not allocated by default;
 * register it only when the collision model relaxes rotational energy.
 */
class FluidRotationalEnergyState final : public FluidState {
public:
    /** @brief Constructs an empty state with no allocation. */
    FluidRotationalEnergyState() = default;

    /**
     * @brief Allocates a rotational-energy buffer sized to the fluid's capacity.
     * @param buffer_size Number of particle slots to allocate.
     */
    ATLAS_HOST explicit FluidRotationalEnergyState(std::size_t buffer_size);

    /**
     * @brief Adopts an existing device buffer as the rotational-energy column.
     * @param rotational_energy Device buffer to take ownership of by move.
     */
    ATLAS_HOST explicit FluidRotationalEnergyState(DeviceBuffer<float> rotational_energy) noexcept;

    /** @copydoc FluidState::size */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /** @copydoc FluidState::compact */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    /** @copydoc FluidState::reset */
    ATLAS_HOST void
    reset() override;

    /** @brief Returns a mutable reference to the owned rotational-energy buffer. */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /** @brief Returns a const reference to the owned rotational-energy buffer. */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _rotational_energy; ///< Per-particle rotational energy, in joules.
};

/**
 * @brief Optional per-particle vibrational-energy column, one @c float per particle, in
 *        joules.
 *
 * One of the three internal-energy modes DSMC can track. Not allocated by default;
 * register it only when the collision model relaxes vibrational energy.
 */
class FluidVibrationalEnergyState final : public FluidState {
public:
    /** @brief Constructs an empty state with no allocation. */
    FluidVibrationalEnergyState() = default;

    /**
     * @brief Allocates a vibrational-energy buffer sized to the fluid's capacity.
     * @param buffer_size Number of particle slots to allocate.
     */
    ATLAS_HOST explicit FluidVibrationalEnergyState(std::size_t buffer_size);

    /**
     * @brief Adopts an existing device buffer as the vibrational-energy column.
     * @param vibrational_energy Device buffer to take ownership of by move.
     */
    ATLAS_HOST explicit FluidVibrationalEnergyState(DeviceBuffer<float> vibrational_energy) noexcept;

    /** @copydoc FluidState::size */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /** @copydoc FluidState::compact */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    /** @copydoc FluidState::reset */
    ATLAS_HOST void
    reset() override;

    /** @brief Returns a mutable reference to the owned vibrational-energy buffer. */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /** @brief Returns a const reference to the owned vibrational-energy buffer. */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _vibrational_energy; ///< Per-particle vibrational energy, in joules.
};

}