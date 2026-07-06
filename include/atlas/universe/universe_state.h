#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/container/container.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/parallel/parallel_fill.h>

#include <cstddef>
#include <utility>

/**
 * @file universe_state.h
 * @brief The per-cell attribute buffers every solver/codec/measurer
 *        reads or writes, held in a `Universe`'s heterogeneous state
 *        store — the per-cell counterpart to `fluid_state.h`'s
 *        per-particle buffers.
 *
 * @details
 * ### Operating principle
 * Same design as `FluidState` (see `fluid_state.h`'s top-of-file
 * documentation): each concrete `Universe*State` owns one
 * `DeviceBuffer<T>` sized to `Universe::cell_count()` and implements the
 * minimal `size`/`reset` interface so `Universe`'s `UniverseStateStore`
 * (a `TypeStore<UniverseState>`) can hold an arbitrary, only-as-needed
 * subset of them. Unlike `FluidState`, no `compact()` exists here —
 * cells are a fixed grid, never added/removed the way particles are.
 *
 * ### Who reads/writes each state (cross-reference)
 * - `UniverseTemperatureState` / `UniverseBulkVelocityState` /
 *   `UniverseThermalEnergyState`: written by `BoltzmannMeasurer`
 *   (equipartition-theorem temperature estimate, see
 *   `boltzmann_measurer.h`); read wherever per-cell macroscopic flow
 *   properties are needed.
 * - `UniverseFieldForceState`: written by `SphSolver` as a per-cell
 *   averaged force diagnostic (see `sph_solver.h`).
 * - `UniverseGravityState`: an external per-cell body-force field (e.g.
 *   consumed by `System::time_integration`).
 * - `UniverseMaxRelativeSpeedState` / `UniverseMaxSigmaGState` /
 *   `UniverseCollisionCountState` / `UniverseCollisionRemainderState`:
 *   the DSMC no-time-counter (NTC) collision-selection bookkeeping — see
 *   `dsmc_solver.h`'s top-of-file documentation for the full derivation
 *   and `DsmcStatistics`/`DsmcSimpleStatistics` for how they're
 *   measured.
 * - `UniverseVolumeState`: per-cell free (fluid-accessible) volume for
 *   cut cells partially occluded by solid geometry — see
 *   `volume_measurer.h`.
 * - `UniverseNumberParticleState`: per-cell particle count, a shared
 *   input to both the NTC formula and `KnudsenCodec::knudsen_number`.
 * - `UniverseKnudsenNumberState`: written by `KnudsenCodec::encode()`,
 *   read by `decode()` (see `knudsen_codec.h`).
 * - `UniverseMaterialRatioState<N>`: per-cell composition (species
 *   volume/number fractions) for an `N`-species mixture; kept as a
 *   header-inline class template (dimension `N` is compile-time, unlike
 *   the non-template states above) since `Container<float, N>` needs a
 *   fixed `N` per instantiation.
 */

namespace atlas {

/**
 * @brief Base interface every per-cell attribute buffer implements, so
 *        `Universe`'s heterogeneous state store can operate on all
 *        registered states uniformly. See this file's top-of-file
 *        documentation.
 */
class UniverseState {
public:
    UniverseState() = default;

    UniverseState(const UniverseState&) = delete;

    UniverseState(UniverseState&&) noexcept = default;

    virtual ~UniverseState() = default;

    UniverseState&
    operator=(const UniverseState&)
        = delete;

    UniverseState&
    operator=(UniverseState&&) noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD virtual std::size_t
    size() const noexcept = 0;

    ATLAS_HOST virtual void
    reset()
        = 0;

protected:
    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_buffer(Buffer& buffer) {
        using value_type = typename Buffer::value_type;
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), value_type {});
    }
};

/** @brief Per-cell temperature; see this file's top-of-file
 *  cross-reference. Every concrete `Universe*State` below follows the
 *  same shape as `FluidPositionState` (see `fluid_state.h`), minus
 *  `compact()`. */
class UniverseTemperatureState final : public UniverseState {
public:
    UniverseTemperatureState() = default;

    ATLAS_HOST explicit UniverseTemperatureState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseTemperatureState(DeviceBuffer<float> temperature) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<float>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _temperature;
};

/** @brief Per-cell mean particle velocity. */
class UniverseBulkVelocityState final : public UniverseState {
public:
    UniverseBulkVelocityState() = default;

    ATLAS_HOST explicit UniverseBulkVelocityState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseBulkVelocityState(DeviceBuffer<Vector3> bulk_velocity) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3> _bulk_velocity;
};

/** @brief Per-cell averaged SPH force diagnostic. */
class UniverseFieldForceState final : public UniverseState {
public:
    UniverseFieldForceState() = default;

    ATLAS_HOST explicit UniverseFieldForceState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseFieldForceState(DeviceBuffer<Vector3> field_force) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3> _field_force;
};

/** @brief Per-cell external body-force (gravity) field. */
class UniverseGravityState final : public UniverseState {
public:
    UniverseGravityState() = default;

    ATLAS_HOST explicit UniverseGravityState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseGravityState(DeviceBuffer<Vector3> gravity) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3> _gravity;
};

/** @brief Per-cell running maximum relative speed sampled among
 *  candidate DSMC pairs; see `dsmc_solver.h`'s NTC derivation. */
class UniverseMaxRelativeSpeedState final : public UniverseState {
public:
    UniverseMaxRelativeSpeedState() = default;

    ATLAS_HOST explicit UniverseMaxRelativeSpeedState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseMaxRelativeSpeedState(DeviceBuffer<float> max_relative_speed) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<float>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _max_relative_speed;
};

/** @brief Per-cell running upper bound on `sigma * g`, the core NTC
 *  collision-rate bound; see `dsmc_solver.h`. */
class UniverseMaxSigmaGState final : public UniverseState {
public:
    UniverseMaxSigmaGState() = default;

    ATLAS_HOST explicit UniverseMaxSigmaGState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseMaxSigmaGState(DeviceBuffer<float> max_sigma_g) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<float>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _max_sigma_g;
};

/** @brief Per-cell free (fluid-accessible) volume; see
 *  `volume_measurer.h`. */
class UniverseVolumeState final : public UniverseState {
public:
    UniverseVolumeState() = default;

    ATLAS_HOST explicit UniverseVolumeState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseVolumeState(DeviceBuffer<float> volume) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<float>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _volume;
};

/** @brief Per-cell summed squared velocity fluctuation, an intermediate
 *  in `BoltzmannMeasurer`'s temperature estimate. */
class UniverseThermalEnergyState final : public UniverseState {
public:
    UniverseThermalEnergyState() = default;

    ATLAS_HOST explicit UniverseThermalEnergyState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseThermalEnergyState(DeviceBuffer<float> thermal_energy) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<float>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _thermal_energy;
};

/** @brief Per-cell particle count; shared input to the NTC formula and
 *  `KnudsenCodec::knudsen_number`. */
class UniverseNumberParticleState final : public UniverseState {
public:
    UniverseNumberParticleState() = default;

    ATLAS_HOST explicit UniverseNumberParticleState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseNumberParticleState(DeviceBuffer<float> number_particle) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<float>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _number_particle;
};

/** @brief Per-cell integer NTC candidate count to draw this step; see
 *  `dsmc_solver.h`. */
class UniverseCollisionCountState final : public UniverseState {
public:
    UniverseCollisionCountState() = default;

    ATLAS_HOST explicit UniverseCollisionCountState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseCollisionCountState(DeviceBuffer<int> collision_count) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<int>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<int>&
    data() const noexcept;

private:
    DeviceBuffer<int> _collision_count;
};

/** @brief Per-cell fractional NTC candidate-count carryover between
 *  steps; see `dsmc_solver.h`. */
class UniverseCollisionRemainderState final : public UniverseState {
public:
    UniverseCollisionRemainderState() = default;

    ATLAS_HOST explicit UniverseCollisionRemainderState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseCollisionRemainderState(DeviceBuffer<float> collision_remainder) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<float>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _collision_remainder;
};

/** @brief Per-cell Knudsen number; written/read by `KnudsenCodec`. */
class UniverseKnudsenNumberState final : public UniverseState {
public:
    UniverseKnudsenNumberState() = default;

    ATLAS_HOST explicit UniverseKnudsenNumberState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseKnudsenNumberState(DeviceBuffer<float> knudsen_number) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<float>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _knudsen_number;
};

/** @brief Per-cell `N`-species composition (`Container<float, N>` per
 *  cell); header-inline since it is a class template (compile-time
 *  species count `N`), unlike the non-template states above. See this
 *  file's top-of-file cross-reference. */
template <std::size_t N>
class UniverseMaterialRatioState final : public UniverseState {
public:
    static_assert(N >= 1, "UniverseMaterialRatioState dimension must be >= 1.");

    using ratio_type = Container<float, N>;

    UniverseMaterialRatioState() = default;

    ATLAS_HOST explicit UniverseMaterialRatioState(const std::size_t cell_count)
        : _material_ratio(cell_count) { }

    ATLAS_HOST explicit UniverseMaterialRatioState(DeviceBuffer<ratio_type> material_ratio) noexcept
        : _material_ratio(std::move(material_ratio)) { }

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override {
        return _material_ratio.size();
    }

    ATLAS_HOST void
    reset() override {
        reset_buffer(_material_ratio);
    }

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<ratio_type>&
    data() noexcept {
        return _material_ratio;
    }

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<ratio_type>&
    data() const noexcept {
        return _material_ratio;
    }

private:
    DeviceBuffer<ratio_type> _material_ratio;
};

}
