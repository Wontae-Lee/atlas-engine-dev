#pragma once

#include <atlas/sampling/sampling.h>

#include <cmath>

namespace atlas::system {

template <typename T>
void
SurfaceSpawnOperator<T>::spawn(DeviceBuffer<Vector3<T>>& particles,
                               const atlas::geometry::QueryOperator<T>& query,
                               const T spacing,
                               T tolerance) const {
    if (!std::isfinite(tolerance) || tolerance < T(0)) tolerance = spacing * T(0.5);

    atlas::sampling::sample_spawn_grid(
        particles,
        query,
        spacing,
        tolerance,
        [] ATLAS_ALL_DEVICE(const atlas::geometry::QueryOperator<T>& query_op,
                            const Vector3<T>& sample,
                            const T tol) {
            return query_op.is_on_surface(sample, tol);
        });
}

template <typename T>
void
VolumeSpawnOperator<T>::spawn(DeviceBuffer<Vector3<T>>& particles,
                              const atlas::geometry::QueryOperator<T>& query,
                              const T spacing,
                              T tolerance) const {
    if (!std::isfinite(tolerance) || tolerance < T(0)) tolerance = T(0);

    atlas::sampling::sample_spawn_grid(
        particles,
        query,
        spacing,
        tolerance,
        [] ATLAS_ALL_DEVICE(const atlas::geometry::QueryOperator<T>& query_op,
                            const Vector3<T>& sample,
                            const T tol) {
            return query_op.is_inside(sample, tol);
        });
}

template <typename T>
void
SpawnOperator<T>::spawn(DeviceBuffer<Vector3<T>>& particles,
                        const atlas::geometry::QueryOperator<T>& query,
                        const T spacing,
                        const T tolerance) const {
    switch (type) {
    case SpawnType::Surface:
        SurfaceSpawnOperator<T> {}.spawn(particles, query, spacing, tolerance);
        return;
    case SpawnType::Volume:
        VolumeSpawnOperator<T> {}.spawn(particles, query, spacing, tolerance);
        return;
    }

    particles.clear();
}

} // namespace atlas::system
