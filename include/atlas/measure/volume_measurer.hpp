#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
VolumeMeasurer<T>::VolumeMeasurer(UniverseHostPtr<T> universe,
                                  DeviceBuffer<Unit<T>> units,
                                  const int samples_per_axis) noexcept
    : Measurer<T>(std::move(universe), nullptr, nullptr)
    , _units(std::move(units))
    , _samples_per_axis(samples_per_axis) {
    ensure_state();
}

template <typename T>
typename VolumeMeasurer<T>::Builder
VolumeMeasurer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
VolumeMeasurer<T>::measure() {
    measure(T(0));
}

template <typename T>
void
VolumeMeasurer<T>::measure(const T dt) {
    if (!this->_universe) {
        return;
    }

    update_units(dt);
    ensure_state();
    measure_volume();
}

template <typename T>
MeasureModeType
VolumeMeasurer<T>::measure_mode() const noexcept {
    return MeasureModeType::Field;
}

template <typename T>
const DeviceBuffer<Unit<T>>&
VolumeMeasurer<T>::units() const noexcept {
    return _units;
}

template <typename T>
int
VolumeMeasurer<T>::samples_per_axis() const noexcept {
    return _samples_per_axis;
}

template <typename T>
void
VolumeMeasurer<T>::ensure_state() {
    if (!this->_universe) {
        return;
    }

    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    if (auto* state = this->_universe->template state<atlas::UniverseVolumeState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<atlas::UniverseVolumeState<T>>(number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }
}

template <typename T>
void
VolumeMeasurer<T>::update_units(const T dt) noexcept {
    if (!(dt > T(0))) {
        return;
    }

    auto* units = atlas::raw_pointer_cast(_units.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });
}

template <typename T>
void
VolumeMeasurer<T>::measure_volume() {
    auto* state = this->_universe->template state<atlas::UniverseVolumeState<T>>();
    if (state == nullptr) {
        return;
    }

    const Vector3<T> lower_corner = this->_universe->lower_corner();
    const Vector3<int> grid_size  = this->_universe->grid_size();
    const T cell_size             = this->_universe->cell_size();
    const T cell_volume           = this->_universe->cell_volume();
    const T inv_cell_size         = this->_universe->inverse_cell_size();
    const int samples_axis        = std::max(1, _samples_per_axis);
    const int samples_per_cell    = samples_axis * samples_axis * samples_axis;
    const T sample_step           = cell_size / static_cast<T>(samples_axis);
    const T sample_volume         = cell_volume / static_cast<T>(samples_per_cell);
    const int num_of_cells        = this->_universe->number_of_cells();

    auto& volume = state->data();
    if (volume.size() != static_cast<std::size_t>(num_of_cells)) {
        volume.resize(static_cast<std::size_t>(num_of_cells));
    }

    if (_units.empty()) {
        auto* volume_ptr = atlas::raw_pointer_cast(volume.data());
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            num_of_cells,
            [=] ATLAS_DEVICE(const int cell) {
                volume_ptr[cell] = cell_volume;
            });
        return;
    }

    _unit_regions.resize(_units.size());
    const Vector3<int> grid_low { 0, 0, 0 };
    const Vector3<int> grid_high = grid_size - Vector3<int> { 1, 1, 1 };
    const atlas::AxisAlignedBoundingBox<T> grid_bound {
        lower_corner,
        lower_corner + grid_size.template cast_to<T>() * cell_size
    };
    const atlas::AxisAlignedBoundingBox<int> grid_index_bound { grid_low, grid_high };
    const Vector3<int> expand { 1, 1, 1 };

    auto* units_ptr         = atlas::raw_pointer_cast(_units.data());
    auto* regions_ptr       = atlas::raw_pointer_cast(_unit_regions.data());
    const auto unit_count   = static_cast<int>(_units.size());
    const int xy_cell_count = grid_size.x * grid_size.y;
    auto* volume_ptr        = atlas::raw_pointer_cast(volume.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        unit_count,
        [=] ATLAS_DEVICE(const int unit_index) {
            const auto& unit = units_ptr[unit_index];
            auto local_bound = unit.geometry_operator().bound();
            auto& region     = regions_ptr[unit_index];
            region.begin     = grid_low;
            region.end       = grid_high;
            region.active    = true;

            const auto world_bound = atlas::transform_aabb(
                local_bound,
                [&unit] ATLAS_DEVICE(const Vector3<T>& point) {
                    return unit.sync_operator().sync_to_world(point);
                });

            if (!world_bound.is_valid()) {
                return;
            }

            if (!world_bound.overlaps(grid_bound)) {
                region.active = false;
                return;
            }

            region.begin = grid_index_bound.clamp(
                atlas::floor((world_bound.lower_corner - lower_corner) * inv_cell_size)
                    .template cast_to<int>()
                - expand);
            region.end = grid_index_bound.clamp(
                atlas::floor((world_bound.upper_corner - lower_corner) * inv_cell_size)
                    .template cast_to<int>()
                + expand);
        });

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell_index) {
            const Vector3<int> cell {
                cell_index % grid_size.x,
                (cell_index / grid_size.x) % grid_size.y,
                cell_index / xy_cell_count
            };
            const Vector3<T> cell_origin = lower_corner + cell.template cast_to<T>() * cell_size;

            int occupied_samples = 0;

            for (int sample = 0; sample < samples_per_cell; ++sample) {
                const Vector3<int> sample_ijk {
                    sample % samples_axis,
                    (sample / samples_axis) % samples_axis,
                    sample / (samples_axis * samples_axis)
                };
                const Vector3<T> sample_point = cell_origin
                    + (sample_ijk.template cast_to<T>() + Vector3<T>(T(0.5))) * sample_step;

                for (int unit_index = 0; unit_index < unit_count; ++unit_index) {
                    const UnitRegion& region = regions_ptr[unit_index];
                    if (!region.contains(cell)) {
                        continue;
                    }

                    const auto& unit             = units_ptr[unit_index];
                    const Vector3<T> local_point = unit.sync_operator().sync_to_local(sample_point);
                    if (unit.geometry_operator().is_inside(local_point, T(0))) {
                        ++occupied_samples;
                        break;
                    }
                }

                if (occupied_samples == samples_per_cell) {
                    break;
                }
            }

            volume_ptr[cell_index] = cell_volume - static_cast<T>(occupied_samples) * sample_volume;
        });
}

template <typename T>
typename VolumeMeasurer<T>::Builder&
VolumeMeasurer<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename VolumeMeasurer<T>::Builder&
VolumeMeasurer<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
    _units = units;
    return *this;
}

template <typename T>
typename VolumeMeasurer<T>::Builder&
VolumeMeasurer<T>::Builder::with_samples_per_axis(const int samples_per_axis) {
    _samples_per_axis = samples_per_axis;
    return *this;
}

template <typename T>
void
VolumeMeasurer<T>::Builder::validate() const {
    atlas::check<std::runtime_error>(static_cast<bool>(_universe))
        << "VolumeMeasurer::Builder: universe must not be null.";
    atlas::check<std::runtime_error>(_samples_per_axis > 0)
        << "VolumeMeasurer::Builder: samples_per_axis must be positive.";
}

template <typename T>
VolumeMeasurer<T>
VolumeMeasurer<T>::Builder::build() const {
    validate();
    return VolumeMeasurer<T>(
        _universe,
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        _samples_per_axis);
}

template <typename T>
atlas::host_shared_ptr<VolumeMeasurer<T>>
VolumeMeasurer<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<VolumeMeasurer<T>>(build());
}

}