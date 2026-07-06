#pragma once

#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>

namespace atlas {

class DsmcSimpleStatistics final : public DsmcStatistics {
public:
    ATLAS_HOST bool
    measure(const DsmcProbe& probe,
            const DeviceBuffer<int>* allocated_solver,
            int index,
            float dt) const override;

public:
    ATLAS_HOST static void
    measure_cells(const DsmcProbe& probe, const int* allocated_solver_ptr, int index, float dt);

private:
};

}
