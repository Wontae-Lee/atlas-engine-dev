// #pragma once
//
// namespace atlas::solver::dsmc::metrics {
// template <typename T>
// T
// GRefRmsOperator<T>::operator()(
//     int cell_id,
//     const system::SpatialHashProbe<T>& neighbor_probe,
//     const system::ParticleDeviceProbe<T>& pdata) const {
//     if (!pdata.vel) {
//         return T(0);
//     }
//
//     const int begin = neighbor_probe.cell_start[cell_id];
//     if (begin < 0) {
//         return T(0);
//     }
//
//     const int end = neighbor_probe.cell_end[cell_id];
//     const int Nc  = end - begin;
//     if (Nc < 2) {
//         return T(0);
//     }
//
//     auto* vel = pdata.vel;
//
//     T sum_v2 = T(0);
//     for (int k = begin; k < end; ++k) {
//         const int p        = neighbor_probe.indices[k];
//         const Vector3<T> v = vel[p];
//         sum_v2             += v.x * v.x + v.y * v.y + v.z * v.z;
//     }
//
//     const T Nc_t    = static_cast<T>(Nc);
//     const T mean_v2 = sum_v2 / Nc_t;
//     if (mean_v2 <= T(0)) {
//         return T(0);
//     }
//
//     const T c_rms = ::sqrt(mean_v2);
//     return factor * c_rms;
// }
// }