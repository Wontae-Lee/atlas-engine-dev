#pragma once

#include <atlas/collect/collect.h>

namespace atlas::system {

template <typename T>
void
SurfaceDespawnOperator<T>::despawn(DeviceBuffer<int>& despawn_indices,
                                   const DeviceBuffer<Vector3<T>>& particles,
                                   const atlas::geometry::QueryOperator<T>& query,
                                   T tolerance) const {
    collect::collect_despawn_indices(
        despawn_indices,
        particles,
        query,
        tolerance,
        [] ATLAS_ALL_DEVICE(const atlas::geometry::QueryOperator<T>& query_op,
                            const Vector3<T>& particle,
                            const T tol) {
            return query_op.is_on_surface(particle, tol);
        });
}

template <typename T>
void
VolumeDespawnOperator<T>::despawn(DeviceBuffer<int>& despawn_indices,
                                  const DeviceBuffer<Vector3<T>>& particles,
                                  const atlas::geometry::QueryOperator<T>& query,
                                  T tolerance) const {
    collect::collect_despawn_indices(
        despawn_indices,
        particles,
        query,
        tolerance,
        [] ATLAS_ALL_DEVICE(const atlas::geometry::QueryOperator<T>& query_op,
                            const Vector3<T>& particle,
                            const T tol) {
            return query_op.is_inside(particle, tol);
        });
}

template <typename T>
void
DespawnOperator<T>::despawn(DeviceBuffer<int>& despawn_indices,
                            const DeviceBuffer<Vector3<T>>& particles,
                            const atlas::geometry::QueryOperator<T>& query,
                            const T tolerance) const {
    switch (type) {
    case DespawnType::Surface:
        SurfaceDespawnOperator<T> {}.despawn(despawn_indices, particles, query, tolerance);
        return;
    case DespawnType::Volume:
        VolumeDespawnOperator<T> {}.despawn(despawn_indices, particles, query, tolerance);
        return;
    }

    despawn_indices.clear();
}

} // namespace atlas::system
