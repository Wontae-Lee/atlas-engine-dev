#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_probe.h>

namespace atlas::system {

template <typename T>
class DsmcStatistics {
public:
    ATLAS_HOST virtual ~DsmcStatistics() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual bool
    measure(const DsmcProbe<T>& probe,
            const DeviceBuffer<int>* allocated_solver,
            int index,
            T dt) const;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DsmcStatistics = atlas::system::DsmcStatistics<T>;

} // namespace atlas

#include <atlas/solver/dsmc/statistics/dsmc_statistics.hpp>
