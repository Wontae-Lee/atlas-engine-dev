#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/math/math.h>
#include <atlas/measure/measurer.h>
#include <atlas/unit/unit.h>

namespace atlas {

template <typename T>
class VolumeMeasurer final : public Measurer<T> {
public:
    class Builder;

private:
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
    VolumeMeasurer() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    VolumeMeasurer(UniverseHostPtr<T> universe,
                   DeviceBuffer<Unit<T>> units,
                   int samples_per_axis = 4) noexcept;

    ~VolumeMeasurer() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure() override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(T dt) override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE MeasureModeType
    measure_mode() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Unit<T>>&
    units() const noexcept;

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
    DeviceBuffer<Unit<T>> _units;

    DeviceBuffer<UnitRegion> _unit_regions;

    int _samples_per_axis { 4 };
};

template <typename T>
class VolumeMeasurer<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_samples_per_axis(int samples_per_axis);

    ATLAS_HOST ATLAS_FORCE_INLINE VolumeMeasurer<T>
    build() const;

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

}

namespace atlas {

template <typename T>
using VolumeMeasurerHostPtr = atlas::host_shared_ptr<atlas::VolumeMeasurer<T>>;

template <typename T>
using VolumeMeasurerDevicePtr = atlas::device_shared_ptr<atlas::VolumeMeasurer<T>>;

}

#include <atlas/measure/volume_measurer.hpp>