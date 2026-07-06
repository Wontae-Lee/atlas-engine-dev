#pragma once

#include <atlas/core/macros.h>
#include <atlas/measure/measurer.h>

namespace atlas {

class BoltzmannMeasurer final : public Measurer {
public:
    class Builder;

public:
    BoltzmannMeasurer() = default;

    ATLAS_HOST
    BoltzmannMeasurer(UniverseHostPtr universe,
                      FluidHostPtr fluid,
                      SearcherHostPtr searcher,
                      MeasureModeType measure_mode = MeasureModeType::all) noexcept;

    ~BoltzmannMeasurer() override = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    measure() override;

    ATLAS_HOST void
    measure(float dt) override;

    ATLAS_NODISCARD ATLAS_HOST MeasureModeType
    measure_mode() const noexcept override;

public:
    ATLAS_HOST void
    measure_field();

    ATLAS_HOST void
    assign_particle_temperature();

private:
    MeasureModeType _measure_mode { MeasureModeType::all };
};

class BoltzmannMeasurer::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_measure_mode(MeasureModeType measure_mode) noexcept;

    ATLAS_NODISCARD ATLAS_HOST BoltzmannMeasurer
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<BoltzmannMeasurer>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    MeasureModeType _measure_mode { MeasureModeType::field };
};

using BoltzmannMeasurerHostPtr = atlas::host_shared_ptr<BoltzmannMeasurer>;

using BoltzmannMeasurerDevicePtr = atlas::device_shared_ptr<BoltzmannMeasurer>;

}
