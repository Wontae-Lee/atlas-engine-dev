#pragma once
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/system/particle_system.h>
namespace atlas::solver::dsmc::metrics {
enum class GRefOpType : int {
    RmsLocal,
    ThermalLocal,
    Constant,
    MaxSpeedLocal,
    AdaptiveRms
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
struct GRefThermalOperator {
    const T* cell_temperature = nullptr;
    T mass                    = T(1);
    T k_b                     = T(1);
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(int cell_id,
               const system::SpatialHashProbe<T>& neighbor_probe,
               const system::ParticleDeviceProbe<T>& pdata) const;
};
template <typename T>
struct GRefConstantOperator {
    T value = T(0);
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(int cell_id,
               const system::SpatialHashProbe<T>& neighbor_probe,
               const system::ParticleDeviceProbe<T>& pdata) const;
};
template <typename T>
struct GRefMaxSpeedOperator {
    T factor = T(1);
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(int cell_id,
               const system::SpatialHashProbe<T>& neighbor_probe,
               const system::ParticleDeviceProbe<T>& pdata) const;
};
template <typename T>
struct GRefAdaptiveRmsOperator {
    const T* prev_g_ref = nullptr;
    T alpha             = T(0.5);
    T factor            = T(1.5);
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
        GRefThermalOperator<T> thermal;
        GRefConstantOperator<T> constant;
        GRefMaxSpeedOperator<T> maxSpeed;
        GRefAdaptiveRmsOperator<T> adaptiveRms;
    };
    GRefOperator() = default;
    ATLAS_HOST explicit GRefOperator(const GRefRmsOperator<T>& op)
        : rms(op) { }
    ATLAS_HOST explicit GRefOperator(const GRefThermalOperator<T>& op)
        : type(GRefOpType::ThermalLocal)
        , thermal(op) { }
    ATLAS_HOST explicit GRefOperator(const GRefConstantOperator<T>& op)
        : type(GRefOpType::Constant)
        , constant(op) { }
    ATLAS_HOST explicit GRefOperator(const GRefMaxSpeedOperator<T>& op)
        : type(GRefOpType::MaxSpeedLocal)
        , maxSpeed(op) { }
    ATLAS_HOST explicit GRefOperator(const GRefAdaptiveRmsOperator<T>& op)
        : type(GRefOpType::AdaptiveRms)
        , adaptiveRms(op) { }
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    compute(int cell_id,
            const system::SpatialHashProbe<T>& probe,
            const system::ParticleDeviceProbe<T>& pdata) const {
        switch (type) {
        case GRefOpType::RmsLocal:
            return rms(cell_id, probe, pdata);
        case GRefOpType::ThermalLocal:
            return thermal(cell_id, probe, pdata);
        case GRefOpType::Constant:
            return constant(cell_id, probe, pdata);
        case GRefOpType::MaxSpeedLocal:
            return maxSpeed(cell_id, probe, pdata);
        case GRefOpType::AdaptiveRms:
            return adaptiveRms(cell_id, probe, pdata);
        default:
            return T(0);
        }
    }
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(int cell_id,
               const system::SpatialHashProbe<T>& neighbor_probe,
               const system::ParticleDeviceProbe<T>& pdata) const {
        return compute(cell_id, neighbor_probe, pdata);
    }
};
}
#include <atlas/solver/dsmc/metrics/g_ref_operator.hpp>