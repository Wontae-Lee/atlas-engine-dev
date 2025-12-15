#pragma once
#include <atlas/memory/memory.h>
#include <atlas/solver/dsmc/dsmc_data.h>
#include <atlas/solver/dsmc/metrics/g_ref_operator.h>

namespace atlas::solver::dsmc {
namespace metrics {
    template <typename T>
    class GRefEstimator {
    public:
        GRefEstimator()          = default;
        virtual ~GRefEstimator() = default;
        ATLAS_HOST ATLAS_FORCE_INLINE virtual GRefOperator<T>
        make_device_operator() const = 0;
        ATLAS_HOST ATLAS_FORCE_INLINE virtual void
        compute(const system::SpatialHashProbe<T>& neighbor_probe,
                const DsmcDeviceProbe<T>& dsmc_probe,
                const system::ParticleDeviceProbe<T>& pdata,
                int n_cells) const = 0;
    };
}

template <typename T>
using GRefEstimator = metrics::GRefEstimator<T>;
template <typename T>
using GRefEstimatorHostPtr = atlas::host_shared_ptr<GRefEstimator<T>>;
template <typename T>
using GRefEstimatorDevicePtr = atlas::device_shared_ptr<GRefEstimator<T>>;
}