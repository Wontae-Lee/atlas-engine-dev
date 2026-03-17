#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <cmath>

namespace atlas::collect {

template <typename T, typename Predicate>
void
collect_despawn_indices(DeviceBuffer<int>& despawn_indices,
                        const DeviceBuffer<Vector3<T>>& particles,
                        const atlas::geometry::QueryOperator<T>& query,
                        T tolerance,
                        Predicate predicate) {
    despawn_indices.clear();

    if (!std::isfinite(tolerance) || tolerance < T(0)) tolerance = T(0);

    const std::size_t particle_count = particles.size();
    if (particle_count == 0) return;

    DeviceBuffer<int> keep_mask(particle_count, 0);
    DeviceBuffer<int> offsets(particle_count, 0);

    const Vector3<T>* particles_ptr = atlas::raw_pointer_cast(particles.data());
    int* keep_mask_ptr              = atlas::raw_pointer_cast(keep_mask.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(particle_count),
        [=] ATLAS_ALL_DEVICE(const int index) {
            keep_mask_ptr[index] = predicate(query, particles_ptr[index], tolerance) ? 1 : 0;
        });

    atlas::exclusive_scan<ExecutionPolicy::device>(
        keep_mask.begin(),
        keep_mask.end(),
        offsets.begin(),
        0);

    const int kept = offsets.back() + keep_mask.back();
    if (kept <= 0) return;

    despawn_indices.resize(static_cast<std::size_t>(kept));

    const int* keep_ptr      = atlas::raw_pointer_cast(keep_mask.data());
    const int* offsets_ptr   = atlas::raw_pointer_cast(offsets.data());
    int* despawn_indices_ptr = atlas::raw_pointer_cast(despawn_indices.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(particle_count),
        [=] ATLAS_ALL_DEVICE(const int index) {
            if (!keep_ptr[index]) return;
            despawn_indices_ptr[offsets_ptr[index]] = index;
        });
}

}
