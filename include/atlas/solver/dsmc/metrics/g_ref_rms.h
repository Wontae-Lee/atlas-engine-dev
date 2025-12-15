#pragma once
#include <atlas/iterator/counting_iterator.h>
#include <atlas/solver/dsmc/metrics/g_ref_estimator.h>
#include <atlas/transform/transform.h>

namespace atlas::solver::dsmc {
namespace metrics {
    template <typename T>
    class GRefRmsEstimator final : public GRefEstimator<T> {
    public:
        GRefRmsEstimator()           = default;
        ~GRefRmsEstimator() override = default;
        ATLAS_HOST ATLAS_FORCE_INLINE GRefOperator<T>

        make_device_operator() const override {
            GRefRmsOperator<T> op;
            return GRefOperator<T>(op);
        }

        ATLAS_HOST ATLAS_FORCE_INLINE void
        compute(
            const system::SpatialHashProbe<T>& neighbor_probe,
            const DsmcDeviceProbe<T>& dsmc_probe,
            const system::ParticleDeviceProbe<T>& pdata,
            int n_cells) const override {
            atlas::counting_iterator<int> cell_begin(0);
            atlas::counting_iterator<int> cell_end(n_cells);
            auto g_ref_operator = _g_ref_rms_operator;
            auto g_ref_per_cell = dsmc_probe.g_ref_per_cell;
            atlas::transform<ExecutionPolicy::device>(
                cell_begin,
                cell_end,
                g_ref_per_cell,
                [neighbor_probe, pdata, g_ref_operator] ATLAS_ALL_DEVICE(int cell_id) {
                    return g_ref_operator(cell_id, neighbor_probe, pdata);
                }
                );
        }

    private:
        GRefRmsOperator<T> _g_ref_rms_operator;
    };
}

template <typename T>
using GRefRmsEstimator = metrics::GRefRmsEstimator<T>;
template <typename T>
using GRefRmsEstimatorHostPtr = atlas::host_shared_ptr<metrics::GRefRmsEstimator<T>>;
template <typename T>
using GRefRmsEstimatorDevicePtr = atlas::device_shared_ptr<metrics::GRefRmsEstimator<T>>;
}