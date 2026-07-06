#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/math/math.h>
#include <atlas/measure/measurer.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

namespace atlas {

class VolumeMeasurer final : public Measurer {
public:
    class Builder;

public:
    struct UnitRegion {
        Int3 begin {};
        Int3 end {};
        bool active {};

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        contains(const Int3& cell) const noexcept {
            return active && atlas::all((cell >= begin) & (cell <= end));
        }
    };

public:
    VolumeMeasurer() = default;

    ATLAS_HOST explicit VolumeMeasurer(UniverseHostPtr universe,
                                       int samples_per_axis = 4) noexcept;

    ~VolumeMeasurer() override = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    measure() override;

    ATLAS_HOST void
    measure(float dt) override;

    ATLAS_NODISCARD ATLAS_HOST MeasureModeType
    measure_mode() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Unit>&
    units() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST int
    samples_per_axis() const noexcept;

private:
    ATLAS_HOST void
    ensure_state();

public:
    ATLAS_HOST void
    update_units(float dt) noexcept;

    ATLAS_HOST void
    measure_volume();

private:
    DeviceBuffer<UnitRegion> _unit_regions;

    int _samples_per_axis { 4 };
};

class VolumeMeasurer::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_samples_per_axis(int samples_per_axis);

    ATLAS_NODISCARD ATLAS_HOST VolumeMeasurer
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<VolumeMeasurer>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    int _samples_per_axis { 4 };
};

using VolumeMeasurerHostPtr = atlas::host_shared_ptr<VolumeMeasurer>;

using VolumeMeasurerDevicePtr = atlas::device_shared_ptr<VolumeMeasurer>;

}
