#pragma once

#include <atlas/atlas.h>

#include <cmath>

ATLAS_FORCE_INLINE int
run_unit_cuda_tests() {
    using scalar_t = float;
    using unit_t   = atlas::Unit<scalar_t>;

    atlas::logger::info() << "atlas_unit_cuda_tests: preparing DeviceBuffer<Unit<float>>";

    atlas::HostBuffer<unit_t> host_units(1);
    host_units[0] = unit_t(
        atlas::GeometryOperator<scalar_t> {},
        atlas::SyncOperator<scalar_t> {},
        atlas::Vector3<scalar_t>(2.0f, 0.0f, 0.0f),
        atlas::Vector3<scalar_t>(0.0f, 0.0f, 0.0f),
        std::nullopt,
        std::nullopt);

    atlas::DeviceBuffer<unit_t> device_units(host_units.begin(), host_units.end());

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        device_units.size(),
        [units = thrust::raw_pointer_cast(device_units.data())] ATLAS_DEVICE(std::size_t i) {
            units[i].update(0.5f);
        });

    host_units = atlas::HostBuffer<unit_t>(device_units.begin(), device_units.end());

    const auto translation = host_units[0].sync_operator().translation;

    atlas::logger::info()
        << "atlas_unit_cuda_tests: update() result translation = ("
        << translation.x << ", "
        << translation.y << ", "
        << translation.z << ")";

    if (std::fabs(translation.x - 1.0f) > 1e-5f) {
        atlas::logger::info() << "atlas_unit_cuda_tests: failed";
        return 1;
    }

    if (std::fabs(translation.y) > 1e-5f || std::fabs(translation.z) > 1e-5f) {
        atlas::logger::info() << "atlas_unit_cuda_tests: failed";
        return 1;
    }

    atlas::logger::info() << "atlas_unit_cuda_tests: success";

    return 0;
}
