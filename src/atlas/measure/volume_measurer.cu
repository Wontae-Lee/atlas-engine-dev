#include <atlas/measure/volume_measurer.h>

#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

VolumeMeasurer::VolumeMeasurer(UniverseHostPtr universe,
                               const int samples_per_axis) noexcept
    : Measurer(std::move(universe), nullptr, nullptr)
    , _samples_per_axis(samples_per_axis) {
    ensure_state();
}

VolumeMeasurer::Builder
VolumeMeasurer::builder() noexcept {
    return Builder {};
}

void
VolumeMeasurer::measure() {
    measure(0.0f);
}

void
VolumeMeasurer::measure(const float dt) {
    if (!_universe) {
        return;
    }

    update_units(dt);
    ensure_state();
    measure_volume();
}

MeasureModeType
VolumeMeasurer::measure_mode() const noexcept {
    return MeasureModeType::field;
}

const DeviceBuffer<Unit>&
VolumeMeasurer::units() const noexcept {
    return _universe->measurer_units().units();
}

int
VolumeMeasurer::samples_per_axis() const noexcept {
    return _samples_per_axis;
}

void
VolumeMeasurer::ensure_state() {
    if (!_universe) {
        return;
    }

    const auto cell_count = static_cast<std::size_t>(_universe->cell_count());

    if (auto* state = _universe->state<atlas::UniverseVolumeState>();
        state == nullptr) {
        _universe->emplace_state<atlas::UniverseVolumeState>(cell_count);
    } else if (state->data().size() != cell_count) {
        state->data().resize(cell_count);
    }
}

void
VolumeMeasurer::update_units(const float dt) noexcept {
    if (!_universe) {
        return;
    }
    _universe->measurer_units().advance(dt);
}

void
VolumeMeasurer::measure_volume() {
    auto* state = _universe->state<atlas::UniverseVolumeState>();
    if (state == nullptr) {
        return;
    }

    const Float3 lower_corner  = _universe->lower_corner();
    const Int3 grid_size       = _universe->grid_size();
    const float cell_size      = _universe->cell_size();
    const float cell_volume    = _universe->cell_volume();
    const float inv_cell_size  = _universe->inverse_cell_size();
    const int samples_axis     = std::max(1, _samples_per_axis);
    const int samples_per_cell = samples_axis * samples_axis * samples_axis;
    const float sample_step    = cell_size / static_cast<float>(samples_axis);
    const float sample_volume  = cell_volume / static_cast<float>(samples_per_cell);
    const int cell_count       = _universe->cell_count();

    auto& volume = state->data();
    if (volume.size() != static_cast<std::size_t>(cell_count)) {
        volume.resize(static_cast<std::size_t>(cell_count));
    }

    if (_universe->measurer_units().empty()) {
        auto* volume_ptr = atlas::raw_pointer_cast(volume.data());
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            cell_count,
            [=] ATLAS_ALL_DEVICE(const int cell) {
                volume_ptr[cell] = cell_volume;
            });
        return;
    }

    _unit_regions.resize(_universe->measurer_units().size());
    const Int3 grid_low(0, 0, 0);
    const Int3 grid_high = grid_size - Int3(1, 1, 1);
    const atlas::AABB grid_bound(
        lower_corner,
        lower_corner + atlas::to_vector3(grid_size) * cell_size);
    const Int3 expand(1, 1, 1);

    auto* units_ptr         = atlas::raw_pointer_cast(_universe->measurer_units().units().data());
    auto* regions_ptr       = atlas::raw_pointer_cast(_unit_regions.data());
    const auto unit_count   = static_cast<int>(_universe->measurer_units().size());
    const int xy_cell_count = grid_size.x * grid_size.y;
    auto* volume_ptr        = atlas::raw_pointer_cast(volume.data());

    // Precompute, per unit, a conservative cell-index range its world-space
    // bound could possibly overlap — lets the per-cell sampling pass below
    // skip units whose bound can't reach a given cell at all, without
    // recomputing world-space bounds per cell.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        unit_count,
        [=] ATLAS_ALL_DEVICE(const int unit_index) {
            const auto& unit       = units_ptr[unit_index];
            const auto local_bound = unit.geometry().bound();
            auto& region           = regions_ptr[unit_index];
            region.begin           = grid_low;
            region.end             = grid_high;
            region.active          = true;

            atlas::AABB world_bound;
            if (local_bound.is_valid()) {
                for (std::size_t corner = 0; corner < 8; ++corner) {
                    world_bound.merge(
                        unit.sync().sync_to_world(local_bound.corner(corner)));
                }
            }

            if (!world_bound.is_valid()) {
                return;
            }

            if (!world_bound.overlaps(grid_bound)) {
                region.active = false;
                return;
            }

            // expand by one cell in each direction as a conservative safety
            // margin (the world_bound -> cell-index conversion is itself an
            // approximation of a possibly non-axis-aligned shape).
            region.begin = atlas::clamp(
                atlas::to_vector3i(
                    atlas::floor((world_bound.lower_corner - lower_corner) * inv_cell_size))
                    - expand,
                grid_low,
                grid_high);
            region.end = atlas::clamp(
                atlas::to_vector3i(
                    atlas::floor((world_bound.upper_corner - lower_corner) * inv_cell_size))
                    + expand,
                grid_low,
                grid_high);
        });

    // Deterministic regular sub-grid sampling (not Monte Carlo/random) of
    // each cell to estimate what fraction of it is occluded by embedded
    // solid geometry; see volume_measurer.h's top-of-file documentation
    // for why free (uncovered) volume matters for downstream density
    // calculations.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell_index) {
            const Int3 cell(
                cell_index % grid_size.x,
                (cell_index / grid_size.x) % grid_size.y,
                cell_index / xy_cell_count);
            const Float3 cell_origin = lower_corner + atlas::to_vector3(cell) * cell_size;

            int occupied_samples = 0;

            for (int sample = 0; sample < samples_per_cell; ++sample) {
                const Int3 sample_ijk(
                    sample % samples_axis,
                    (sample / samples_axis) % samples_axis,
                    sample / (samples_axis * samples_axis));
                const Float3 sample_point = cell_origin
                    + (atlas::to_vector3(sample_ijk) + Float3(0.5f, 0.5f, 0.5f)) * sample_step;

                for (int unit_index = 0; unit_index < unit_count; ++unit_index) {
                    const UnitRegion& region = regions_ptr[unit_index];
                    if (!region.contains(cell)) {
                        continue;
                    }

                    const auto& unit         = units_ptr[unit_index];
                    const Float3 local_point = unit.sync().sync_to_local(sample_point);
                    if (unit.geometry().is_inside(local_point, 0.0f)) {
                        ++occupied_samples;
                        break;
                    }
                }

                // Early-exit once every sample so far is occluded — no
                // remaining sample can push occupied_samples any higher, so
                // finishing the loop would be wasted work.
                if (occupied_samples == samples_per_cell) {
                    break;
                }
            }

            volume_ptr[cell_index] = cell_volume - static_cast<float>(occupied_samples) * sample_volume;
        });
}

VolumeMeasurer::Builder&
VolumeMeasurer::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

VolumeMeasurer::Builder&
VolumeMeasurer::Builder::with_samples_per_axis(const int samples_per_axis) {
    _samples_per_axis = samples_per_axis;
    return *this;
}

void
VolumeMeasurer::Builder::validate() const {
    atlas::check<std::runtime_error>(static_cast<bool>(_universe))
        << "VolumeMeasurer::Builder: universe must not be null.";
    atlas::check<std::runtime_error>(_samples_per_axis > 0)
        << "VolumeMeasurer::Builder: samples_per_axis must be positive.";
}

VolumeMeasurer
VolumeMeasurer::Builder::build() const {
    validate();
    return VolumeMeasurer(_universe, _samples_per_axis);
}

atlas::host_shared_ptr<VolumeMeasurer>
VolumeMeasurer::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<VolumeMeasurer>(build());
}

}
