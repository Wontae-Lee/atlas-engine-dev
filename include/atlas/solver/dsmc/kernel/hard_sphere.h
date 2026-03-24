// #pragma once
//
// #include <atlas/searcher/spatial_hashing_searcher.h>
// #include <atlas/solver/dsmc/kernel/collide_kernel.h>
//
// namespace atlas::solver::dsmc {
// namespace kernel {
//     template <typename T>
//     class HardSphereKernel : public CollideKernel<T> {
//     public:
//         HardSphereKernel()           = default;
//         ~HardSphereKernel() override = default;
//
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         operator()(const system::ParticleDeviceProbe<T>& data,
//                    const DsmcDeviceProbe<T>& dsmc_probe,
//                    const system::SpatialHashProbe<T>& neighbor_probe,
//                    DeviceBuffer<int>& d_flattened_collision) const override;
//
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         collide(const system::ParticleDeviceProbe<T>& data,
//                 const DsmcDeviceProbe<T>& dsmc_probe,
//                 const system::SpatialHashProbe<T>& neighbor_probe,
//                 DeviceBuffer<int>& d_flattened_collision) const override;
//     };
// }
//
// template <typename T>
// using HardSphereKernel = kernel::HardSphereKernel<T>;
// template <typename T>
// using HardSphereKernelHostPtr = atlas::host_shared_ptr<kernel::HardSphereKernel<T>>;
// template <typename T>
// using HardSphereKernelDevicePtr = atlas::device_shared_ptr<kernel::HardSphereKernel<T>>;
// }
//
// #include <atlas/solver/dsmc/kernel/hard_sphere.hpp>