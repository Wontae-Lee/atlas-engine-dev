#include <atlas/source/volume_source.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/geometry/geometry.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/sync/sync.h>

#include <algorithm>
#include <atlas/buffer/host_buffer.h>
#include <stdexcept>
#include <utility>

namespace atlas {

VolumeSource::VolumeSource(Unit unit, const float tolerance, const float spacing)
    : _unit(std::move(unit))
    , _tolerance(tolerance)
    , _spacing(spacing) {

    const Geometry& geometry = _unit.geometry();
    const AABB bound         = geometry.bound();

    // No sampleable region (degenerate bound) or no grid step: leave the cache empty.
    if (!bound.is_valid() || !(spacing > 0.0f)) {
        return;
    }

    const Float3 lower = bound.lower_corner;
    const Float3 upper = bound.upper_corner;

    const int nx = atlas::sample_axis_count(lower.x, upper.x, spacing);
    const int ny = atlas::sample_axis_count(lower.y, upper.y, spacing);
    const int nz = atlas::sample_axis_count(lower.z, upper.z, spacing);

    // Accumulate on the host first: the accepted count is unknown up front, so
    // grow a host vector and upload it to the device in one shot at the end.
    HostBuffer<Float3> local_positions;

    // Walk the bound as a regular local-space grid; keep points inside the volume.
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int iz = 0; iz < nz; ++iz) {
                const Float3 sample(
                    lower.x + static_cast<float>(ix) * spacing,
                    lower.y + static_cast<float>(iy) * spacing,
                    lower.z + static_cast<float>(iz) * spacing);

                if (geometry.is_inside(sample, tolerance)) {
                    local_positions.push_back(sample);
                }
            }
        }
    }

    // Single host→device upload of the accepted local points.
    _cache = DeviceBuffer<Float3>(local_positions.begin(), local_positions.end());
}

VolumeSource::Builder
VolumeSource::builder() noexcept {
    return Builder {};
}

int
VolumeSource::spawn(FluidPositionState* positions, const std::size_t offset) const {
    // Nothing to do without a destination or any cached points.
    if (positions == nullptr || _cache.empty()) {
        return 0;
    }

    DeviceBuffer<Float3>& out = positions->data();

    // The write window starts at `offset`; if it is past the end, there is no room.
    if (offset >= out.size()) {
        return 0;
    }

    // Emit at most the cached count, clamped to the free tail of the buffer.
    const std::size_t writable = std::min(_cache.size(), out.size() - offset);

    if (writable == 0) {
        return 0;
    }

    // Snapshot the current pose by value so the device lambda captures a plain
    // Sync (host+device, trivially copyable) rather than referencing the unit.
    const Sync sync         = _unit.sync();
    const Float3* cache_ptr = atlas::raw_pointer_cast(_cache.data());
    Float3* out_ptr         = atlas::raw_pointer_cast(out.data());

    // Device pass: transform each cached local point to world space into the tail.
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        writable,
        [=] ATLAS_ALL_DEVICE(const std::size_t i) {
            out_ptr[offset + i] = sync.sync_to_world(cache_ptr[i]);
        });

    return static_cast<int>(writable);
}

VolumeSource::Builder&
VolumeSource::Builder::with_unit(Unit unit) {
    _unit = std::move(unit);
    return *this;
}

VolumeSource::Builder&
VolumeSource::Builder::with_tolerance(const float tolerance) noexcept {
    _tolerance = tolerance;
    return *this;
}

VolumeSource::Builder&
VolumeSource::Builder::with_spacing(const float spacing) noexcept {
    _spacing = spacing;
    return *this;
}

VolumeSource
VolumeSource::Builder::build() {
    validate();

    // Construction here runs the one-time cache build (grid sample + inside test).
    VolumeSource source(std::move(*_unit), _tolerance, _spacing);

    // Reset so the builder can be reused for another source.
    _unit.reset();
    _tolerance = 0.0f;
    _spacing   = 0.1f;

    return source;
}

atlas::host_shared_ptr<VolumeSource>
VolumeSource::Builder::make_host_shared() {
    return atlas::make_host_shared<VolumeSource>(build());
}

void
VolumeSource::Builder::validate() const {
    if (!_unit) {
        throw std::runtime_error("VolumeSource::Builder: unit must not be null.");
    }

    if (!atlas::isfinite(_tolerance) || _tolerance < 0.0f) {
        throw std::runtime_error("VolumeSource::Builder: tolerance must be finite and non-negative.");
    }

    if (!atlas::isfinite(_spacing) || !(_spacing > 0.0f)) {
        throw std::runtime_error("VolumeSource::Builder: spacing must be finite and positive.");
    }
}

}
