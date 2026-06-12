#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/spatial/transformed_bounds.h>
#include <atlas/transform/transform_reduce.h>

namespace atlas::detail {

template <typename T>
ColliderBoundCache<T>::ColliderBoundCache(const std::size_t unit_count)
    : _unit_bounds(unit_count) {
}

template <typename T>
void
ColliderBoundCache<T>::refresh(const DeviceBuffer<Unit<T>>& units) {
    if (_unit_bounds.size() != units.size()) {
        _unit_bounds.resize(units.size());
    }

    auto* units_ptr = atlas::raw_pointer_cast(units.data());
    auto* bounds_ptr = atlas::raw_pointer_cast(_unit_bounds.data());
    const int unit_count = static_cast<int>(units.size());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        unit_count,
        [units_ptr, bounds_ptr] ATLAS_DEVICE(const int unit_index) {
            const auto& unit = units_ptr[unit_index];
            auto local_bound = unit.geometry_operator().bound();
            auto& world_bound = bounds_ptr[unit_index];
            const auto transformed_bound = atlas::transform_aabb(
                local_bound,
                [&unit] ATLAS_DEVICE(const Vector3<T>& point) {
                    return unit.sync_operator().sync_to_world(point);
                });
            world_bound.lower_corner = transformed_bound.lower_corner;
            world_bound.upper_corner = transformed_bound.upper_corner;
        });

    _scene_bound = atlas::transform_reduce<ExecutionPolicy::device>(
        _unit_bounds.begin(),
        _unit_bounds.end(),
        Bound {},
        [] ATLAS_DEVICE(const Bound& bound) {
            return bound.is_valid() ? bound : Bound {};
        },
        [] ATLAS_DEVICE(Bound lhs, const Bound& rhs) {
            lhs.merge(rhs);
            return lhs;
        });

    _covers_units = atlas::transform_reduce<ExecutionPolicy::device>(
        _unit_bounds.begin(),
        _unit_bounds.end(),
        true,
        [] ATLAS_DEVICE(const Bound& bound) {
            return bound.is_valid();
        },
        [] ATLAS_DEVICE(const bool lhs, const bool rhs) {
            return lhs && rhs;
        });
}

template <typename T>
const DeviceBuffer<typename ColliderBoundCache<T>::Bound>&
ColliderBoundCache<T>::unit_bounds() const noexcept {
    return _unit_bounds;
}

template <typename T>
const typename ColliderBoundCache<T>::Bound&
ColliderBoundCache<T>::scene_bound() const noexcept {
    return _scene_bound;
}

template <typename T>
bool
ColliderBoundCache<T>::covers_units() const noexcept {
    return _covers_units;
}

} // namespace atlas::detail
