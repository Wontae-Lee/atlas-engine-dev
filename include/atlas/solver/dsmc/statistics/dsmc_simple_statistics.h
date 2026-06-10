#pragma once

#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>

namespace atlas::system {

template <typename T>
class DsmcSimpleStatistics final : public DsmcStatistics<T> {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure(const DsmcProbe<T>& probe,
            const DeviceBuffer<int>* allocated_solver,
            int index,
            T dt) const override;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DsmcSimpleStatistics = atlas::system::DsmcSimpleStatistics<T>;

} // namespace atlas

#include <atlas/solver/dsmc/statistics/dsmc_simple_statistics.hpp>
