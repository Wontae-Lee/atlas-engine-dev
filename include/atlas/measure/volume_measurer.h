#pragma once

/**
 * @file volume_measurer.h
 * @brief Declares a measurer that stores per-cell available volume.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/math/math.h>
#include <atlas/measure/measurer.h>
#include <atlas/unit/unit.h>

namespace atlas {

/**
 * @brief Measures cell-wise free volume after subtracting solid unit occupancy.
 *
 * `VolumeMeasurer` updates the configured units by the supplied time step, then
 * estimates how much of each universe cell is occupied by those units. The
 * remaining free volume is written to `UniverseVolumeState<T>`.
 *
 * Occupancy is estimated by regular sub-cell sampling. Each sample point is
 * transformed from world space into the unit's local space through the unit sync
 * operator before calling the unit geometry operator. This makes translation and
 * rotation part of the measurement.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class VolumeMeasurer final : public Measurer<T> {
public:
    /**
     * @brief Fluent builder for validated VolumeMeasurer construction.
     */
    class Builder;

private:
    /**
     * @brief Cell range overlapped by one unit's world-space broad-phase bounds.
     */
    struct UnitRegion {
        Vector3<int> begin {};
        Vector3<int> end {};
        bool active {};

        ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        contains(const Vector3<int>& cell) const noexcept {
            return active && atlas::all((cell >= begin) & (cell <= end));
        }
    };

public:
    /**
     * @brief Constructs an empty measurer.
     */
    VolumeMeasurer() = default;

    /**
     * @brief Constructs a volume measurer from a universe and solid units.
     *
     * @param universe Universe receiving `UniverseVolumeState<T>`.
     * @param units Solid units that occupy volume inside universe cells.
     * @param samples_per_axis Number of regular samples along each cell axis.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    VolumeMeasurer(UniverseHostPtr<T> universe,
                   DeviceBuffer<Unit<T>> units,
                   int samples_per_axis = 4) noexcept;

    /**
     * @brief Default destructor.
     */
    ~VolumeMeasurer() override = default;

    /**
     * @brief Creates an empty builder.
     *
     * @return Builder object used for chained construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Measures volume without advancing units.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure() override;

    /**
     * @brief Advances units by @p dt and measures cell-wise available volume.
     *
     * @param dt Simulation time step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(T dt) override;

    /**
     * @brief Returns the measurement mode.
     *
     * @return `MeasureModeType::Field`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE MeasureModeType
    measure_mode() const noexcept override;

    /**
     * @brief Returns read-only access to measured units.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Unit<T>>&
    units() const noexcept;

    /**
     * @brief Returns the number of samples used along each cell axis.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    samples_per_axis() const noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_state();

public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_units(T dt) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure_volume();

private:
    /**
     * @brief Solid units used as volume-occupying geometry.
     */
    DeviceBuffer<Unit<T>> _units;

    /**
     * @brief Reused per-unit broad-phase cell ranges.
     */
    DeviceBuffer<UnitRegion> _unit_regions;

    /**
     * @brief Number of regular samples along each cell axis.
     */
    int _samples_per_axis { 4 };
};

/**
 * @brief Fluent builder for VolumeMeasurer.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class VolumeMeasurer<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the target universe.
     *
     * @param universe Universe receiving `UniverseVolumeState<T>`.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the solid units used for volume occupancy.
     *
     * @param units Units copied into the constructed measurer.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Sets the regular sampling resolution per cell axis.
     *
     * @param samples_per_axis Number of samples along each axis. Values below 1 are rejected.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_samples_per_axis(int samples_per_axis);

    /**
     * @brief Builds a validated volume measurer value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE VolumeMeasurer<T>
    build() const;

    /**
     * @brief Builds a validated host shared pointer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<VolumeMeasurer<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};
    HostBuffer<Unit<T>> _units;
    int _samples_per_axis { 4 };
};

} // namespace atlas

namespace atlas {


/**
 * @brief Host shared-pointer alias for VolumeMeasurer.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using VolumeMeasurerHostPtr = atlas::host_shared_ptr<atlas::VolumeMeasurer<T>>;

/**
 * @brief Device shared-pointer alias for VolumeMeasurer.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using VolumeMeasurerDevicePtr = atlas::device_shared_ptr<atlas::VolumeMeasurer<T>>;

} // namespace atlas

#include <atlas/measure/volume_measurer.hpp>
