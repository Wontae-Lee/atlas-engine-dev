#pragma once
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/system/particle_system.h>

namespace atlas::solver::dsmc::metrics {
enum class GRefOpType : int {
    RmsLocal,
};

template <typename T>
struct GRefRmsOperator {
    T factor = T(1.5);
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(int cell_id,
               const system::SpatialHashProbe<T>& neighbor_probe,
               const system::ParticleDeviceProbe<T>& pdata) const;
};

template <typename T>
struct GRefOperator {
    GRefOpType type = GRefOpType::RmsLocal;

    union {
        GRefRmsOperator<T> rms;
    };

    GRefOperator() = default;

    ATLAS_HOST explicit
    GRefOperator(const GRefRmsOperator<T>& op)
        : rms(op) {}

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    compute(int cell_id,
            const system::SpatialHashProbe<T>& probe,
            const system::ParticleDeviceProbe<T>& pdata) const {
        switch (type) {
        case GRefOpType::RmsLocal:
            return rms(cell_id, probe, pdata);
        default:
            return T(0);
        }
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(const int cell_id,
               const system::SpatialHashProbe<T>& neighbor_probe,
               const system::ParticleDeviceProbe<T>& pdata) const {
        return compute(cell_id, neighbor_probe, pdata);
    }
};
}

#include <atlas/solver/dsmc/metrics/g_ref_operator.hpp>