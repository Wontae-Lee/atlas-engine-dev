#pragma once
#include <atlas/iterator/counting_iterator.h>
#include <atlas/solver/dsmc/metrics/g_ref_estimator.h>
#include <atlas/solver/dsmc/metrics/g_ref_operator.h>
#include <atlas/transform/transform.h>
namespace atlas::solver::dsmc {
namespace metrics {
    template <typename T>
    class GRefMaxSpeedEstimator final : public GRefEstimator<T> {
    public:
        GRefMaxSpeedEstimator()           = default;
        ~GRefMaxSpeedEstimator() override = default;
        ATLAS_HOST ATLAS_FORCE_INLINE GRefOperator<T>
        make_device_operator() const override {
            GRefMaxSpeedOperator<T> op = _g_ref_max_speed_operator;
            return GRefOperator<T>(op);
        }
        ATLAS_HOST ATLAS_FORCE_INLINE void
        compute(const system::SpatialHashProbe<T>& neighbor_probe,
                const DsmcDeviceProbe<T>& dsmc_probe,
                const system::ParticleDeviceProbe<T>& pdata,
                int n_cells) const override {
            atlas::counting_iterator<int> cell_begin(0);
            atlas::counting_iterator<int> cell_end(n_cells);
            auto g_ref_operator = _g_ref_max_speed_operator;
            auto g_ref_per_cell = dsmc_probe.g_ref_per_cell;
            atlas::transform<ExecutionPolicy::device>(
                cell_begin,
                cell_end,
                g_ref_per_cell,
                [neighbor_probe, pdata, g_ref_operator] ATLAS_ALL_DEVICE(int cell_id) {
                    return g_ref_operator(cell_id, neighbor_probe, pdata);
                });
        }

    private:
        GRefMaxSpeedOperator<T> _g_ref_max_speed_operator;
    };
}
template <typename T>
using GRefMaxSpeedEstimator = metrics::GRefMaxSpeedEstimator<T>;
template <typename T>
using GRefMaxSpeedEstimatorHostPtr = atlas::host_shared_ptr<metrics::GRefMaxSpeedEstimator<T>>;
template <typename T>
using GRefMaxSpeedEstimatorDevicePtr = atlas::device_shared_ptr<metrics::GRefMaxSpeedEstimator<T>>;
}