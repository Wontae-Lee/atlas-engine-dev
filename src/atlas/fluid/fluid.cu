#include <atlas/fluid/fluid.h>

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <stdexcept>
#include <utility>

namespace atlas {

Fluid::Fluid(const std::size_t buffer_size)
    : _buffer_size(buffer_size)
    , _active(buffer_size) {
    _states.reserve(3);
    emplace_state<FluidPositionState>(buffer_size);
    emplace_state<FluidVelocityState>(buffer_size);
    emplace_state<FluidSpeciesState>(buffer_size);
}

Fluid::Fluid(const std::size_t buffer_size, MaterialDictionaryHostPtr materials)
    : Fluid(buffer_size) {
    _materials = std::move(materials);
}

Fluid::Builder
Fluid::builder() noexcept {
    return Builder {};
}

void
Fluid::set_particle_count(const std::size_t particle_count) {
    if (particle_count > _buffer_size) {
        throw std::out_of_range("Fluid::set_particle_count: particle_count exceeds buffer_size.");
    }
    _particle_count = particle_count;
}

DeviceBuffer<int>&
Fluid::active() noexcept {
    return _active;
}

const DeviceBuffer<int>&
Fluid::active() const noexcept {
    return _active;
}

std::size_t
Fluid::compact() {
    const auto alive = _particle_count;

    if (alive == 0 || _active.size() < alive) {
        return alive;
    }

    const int particle_count = static_cast<int>(alive);

    if (_survivor_offsets.size() != alive) {
        _survivor_offsets.resize(alive);
    }

    // A survivor's prefix sum is its slot in the compacted buffers.
    const auto* flags = atlas::raw_pointer_cast(_active.data());

    atlas::exclusive_scan<ExecutionPolicy::device>(
        flags,
        flags + particle_count,
        _survivor_offsets.begin(),
        0);

    if (_survivor_total.size() < 1) {
        _survivor_total.resize(1);
    }

    auto* total         = atlas::raw_pointer_cast(_survivor_total.data());
    const auto* offsets = atlas::raw_pointer_cast(_survivor_offsets.data());
    const int last      = particle_count - 1;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        1,
        [=] ATLAS_ALL_DEVICE(int) {
            total[0] = offsets[last] + flags[last];
        });

    int survivors = 0;
    atlas::copy_device_to_host(total, &survivors, 1);

    const auto kept = static_cast<std::size_t>(survivors < 0 ? 0 : survivors);

    if (kept == alive) {
        return kept;
    }

    if (kept > 0) {
        if (_compact_indices.size() != kept) {
            _compact_indices.resize(kept);
        }

        auto* compact_indices = atlas::raw_pointer_cast(_compact_indices.data());

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            particle_count,
            [=] ATLAS_ALL_DEVICE(const int i) {
                if (flags[i] != 0) {
                    compact_indices[offsets[i]] = static_cast<std::size_t>(i);
                }
            });
    }

    // Every state holds one entry per particle, so they all gather by the same
    // indices.
    for (auto& [type, fluid_state] : _states) {
        fluid_state->compact(_compact_indices, kept);
    }

    // Whatever survived is alive by definition, so the flags need gathering no
    // more than a fill does.
    atlas::parallel_fill<ExecutionPolicy::device>(
        _active.begin(),
        _active.begin() + static_cast<std::ptrdiff_t>(kept),
        1);

    set_particle_count(kept);

    return kept;
}

FluidStateStore&
Fluid::states() noexcept {
    return _states;
}

const FluidStateStore&
Fluid::states() const noexcept {
    return _states;
}

std::size_t
Fluid::buffer_size() const noexcept {
    return _buffer_size;
}

std::size_t
Fluid::particle_count() const noexcept {
    return _particle_count;
}

float
Fluid::statistical_weight() const noexcept {
    return _statistical_weight;
}

const MaterialDictionaryHostPtr&
Fluid::materials() const noexcept {
    return _materials;
}

Fluid::Builder&
Fluid::Builder::with_buffer_size(const std::size_t buffer_size) noexcept {
    _buffer_size = buffer_size;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_particle_count(const std::size_t particle_count) noexcept {
    _particle_count = particle_count;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_statistical_weight(const float statistical_weight) noexcept {
    _statistical_weight = statistical_weight;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_materials(MaterialDictionaryHostPtr materials) noexcept {
    _materials = std::move(materials);
    return *this;
}

void
Fluid::Builder::validate() const {
    if (!(_statistical_weight > 0.0f)) {
        throw std::runtime_error(
            "Fluid::Builder: statistical_weight must be positive.");
    }
    if (_particle_count > _buffer_size) {
        throw std::runtime_error(
            "Fluid::Builder: particle_count exceeds buffer_size.");
    }
}

Fluid
Fluid::Builder::build() const {
    validate();

    Fluid fluid(_buffer_size, _materials);
    fluid._statistical_weight = _statistical_weight;
    fluid.set_particle_count(_particle_count);

    return fluid;
}

atlas::host_unique_ptr<Fluid>
Fluid::Builder::make_host_unique() const {
    return atlas::make_host_unique<Fluid>(build());
}

}
