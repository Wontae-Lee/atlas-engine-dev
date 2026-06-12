#pragma once

namespace atlas::detail {

template <typename T>
struct SinkRefreshUnitBound final {
    const Unit<T>* units {};
    atlas::AxisAlignedBoundingBox<T>* bounds {};
    T expand {};

    ATLAS_DEVICE void
    operator()(const int unit_index) const {
        const auto& unit = units[unit_index];
        const auto local_bound = unit.geometry_operator().bound();
        auto& world_bound = bounds[unit_index];
        const auto transformed_bound = atlas::transform_aabb(
            local_bound,
            [&unit] ATLAS_DEVICE(const Vector3<T>& point) {
                return unit.sync_operator().sync_to_world(point);
            });

        world_bound.lower_corner = transformed_bound.lower_corner;
        world_bound.upper_corner = transformed_bound.upper_corner;

        if (expand > T(0)) {
            world_bound.expand(expand);
        }
    }
};

template <typename T>
void
SinkUnitBounds<T>::refresh(const DeviceBuffer<Unit<T>>& units,
                           DeviceBuffer<atlas::AxisAlignedBoundingBox<T>>& unit_bounds,
                           const T tolerance) const noexcept {
    if (units.empty()) {
        unit_bounds.clear();
        return;
    }

    if (unit_bounds.size() != units.size()) {
        unit_bounds.resize(units.size());
    }

    auto* units_ptr = atlas::raw_pointer_cast(units.data());
    auto* bounds_ptr = atlas::raw_pointer_cast(unit_bounds.data());
    const int unit_count = static_cast<int>(units.size());
    const T expand = tolerance > T(0) ? tolerance : T(0);

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        unit_count,
        SinkRefreshUnitBound<T> {
            units_ptr,
            bounds_ptr,
            expand
        });
}

} // namespace atlas::detail
