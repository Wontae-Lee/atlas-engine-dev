#pragma once

#include <atlas/core/macros.h>
#include <atlas/measure/measurer.h>

namespace atlas {

template <typename T>
class BoltzmanMeasurer final : public Measurer<T> {
public:
    class Builder;

public:
    BoltzmanMeasurer() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    BoltzmanMeasurer(UniverseHostPtr<T> universe,
                     FluidHostPtr<T> fluid,
                     SpatialHashingSearcherHostPtr<T> searcher,
                     MeasureModeType measure_mode = MeasureModeType::All) noexcept;

    ~BoltzmanMeasurer() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure() override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(T dt) override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE MeasureModeType
    measure_mode() const noexcept override;

private:
    MeasureModeType _measure_mode { MeasureModeType::All };
};

template <typename T>
class BoltzmanMeasurer<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measure_mode(MeasureModeType measure_mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE BoltzmanMeasurer<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<BoltzmanMeasurer<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SpatialHashingSearcherHostPtr<T> _searcher {};

    MeasureModeType _measure_mode { MeasureModeType::Field };
};

}

namespace atlas {

template <typename T>
using BoltzmanMeasurerHostPtr = atlas::host_shared_ptr<atlas::BoltzmanMeasurer<T>>;

template <typename T>
using BoltzmanMeasurerDevicePtr = atlas::device_shared_ptr<atlas::BoltzmanMeasurer<T>>;

}

#include <atlas/measure/boltzman_measurer.hpp>