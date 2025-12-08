#pragma once
namespace atlas::solver::dsmc::metrics {
template <typename T>
T
GRefRmsOperator<T>::operator()(
    int cell_id,
    const system::SpatialHashProbe<T>& neighbor_probe,
    const system::ParticleDeviceProbe<T>& pdata) const {
    if (!pdata.vel) {
        return T(0);
    }
    int begin = neighbor_probe.cell_start[cell_id];
    if (begin < 0) {
        return T(0);
    }
    int end = neighbor_probe.cell_end[cell_id];
    int Nc  = end - begin;
    if (Nc < 2) {
        return T(0);
    }
    Vector3<T>* vel = pdata.vel;
    T sum_v2        = T(0);
    for (int k = begin; k < end; ++k) {
        int p        = neighbor_probe.indices[k];
        Vector3<T> v = vel[p];
        sum_v2 += v.x * v.x + v.y * v.y + v.z * v.z;
    }
    T mean_v2 = sum_v2 / static_cast<T>(Nc);
    if (mean_v2 <= T(0)) {
        return T(0);
    }
    T c_rms = ::sqrt(mean_v2);
    return factor * c_rms;
}
template <typename T>
T
GRefThermalOperator<T>::operator()(
    int cell_id,
    const system::SpatialHashProbe<T>&,
    const system::ParticleDeviceProbe<T>&) const {
    if (!cell_temperature) {
        return T(0);
    }
    T Tcell = cell_temperature[cell_id];
    if (Tcell <= T(0)) {
        return T(0);
    }
    const T pi = static_cast<T>(3.14159265358979323846);
    T val      = (T(16) * k_b * Tcell) / (pi * mass);
    if (val <= T(0)) {
        return T(0);
    }
    return ::sqrt(val);
}
template <typename T>
T
GRefConstantOperator<T>::operator()(
    int,
    const system::SpatialHashProbe<T>&,
    const system::ParticleDeviceProbe<T>&) const {
    return value;
}
template <typename T>
T
GRefMaxSpeedOperator<T>::operator()(
    int cell_id,
    const system::SpatialHashProbe<T>& neighbor_probe,
    const system::ParticleDeviceProbe<T>& pdata) const {
    using Vec3 = Vector3<T>;
    if (!pdata.vel) {
        return T(0);
    }
    int begin = neighbor_probe.cell_start[cell_id];
    if (begin < 0) {
        return T(0);
    }
    int end = neighbor_probe.cell_end[cell_id];
    int Nc  = end - begin;
    if (Nc < 2) {
        return T(0);
    }
    Vec3* vel = pdata.vel;
    T vmax2   = T(0);
    for (int k = begin; k < end; ++k) {
        int p  = neighbor_probe.indices[k];
        Vec3 v = vel[p];
        T v2   = v.x * v.x + v.y * v.y + v.z * v.z;
        if (v2 > vmax2) {
            vmax2 = v2;
        }
    }
    if (vmax2 <= T(0)) {
        return T(0);
    }
    T vmax = ::sqrt(vmax2);
    return factor * T(2) * vmax;
}
template <typename T>
T
GRefAdaptiveRmsOperator<T>::operator()(
    int cell_id,
    const system::SpatialHashProbe<T>& neighbor_probe,
    const system::ParticleDeviceProbe<T>& pdata) const {
    GRefRmsOperator<T> rmsOp;
    rmsOp.factor = factor;
    T g_rms      = rmsOp(cell_id, neighbor_probe, pdata);
    if (!prev_g_ref) {
        return g_rms;
    }
    T g_old = prev_g_ref[cell_id];
    if (g_old <= T(0)) {
        return g_rms;
    }
    T oneMinusAlpha = T(1) - alpha;
    return oneMinusAlpha * g_old + alpha * g_rms;
}
}