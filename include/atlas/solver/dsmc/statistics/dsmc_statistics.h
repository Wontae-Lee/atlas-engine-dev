#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_probe.h>

namespace atlas {

class DsmcStatistics {
public:
    ATLAS_HOST virtual ~DsmcStatistics() = default;

    ATLAS_HOST virtual bool
    measure(const DsmcProbe& probe,
            const DeviceBuffer<int>* allocated_solver,
            int index,
            float dt) const;

public:
    ATLAS_HOST static void
    measure_cells(const DsmcProbe& probe, const int* allocated_solver_ptr, int index, float dt);

private:
};

}
