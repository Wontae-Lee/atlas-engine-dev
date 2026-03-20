#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Source<T>::Source(Unit<T> unit,
                  const SpawnType spawn_type,
                  const T spacing,
                  const T tolerance) noexcept
    : _unit(std::move(unit))
    , _spawn_operator(spawn_type)
    , _spacing(spacing)
    , _tolerance(tolerance) {
    update_species_distribution();
}

template <typename T>
typename Source<T>::Builder
Source<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Source<T>::emit(ParticleDeviceProbe<T>& particle_probe) {
    if (!particle_probe.valid()) return;

    // Phase 1: Generate and cache local positions if needed
    if (_local_positions_cache.empty()) {
        _spawn_operator.spawn(
            _local_positions_cache,
            _unit.query_operator(),
            _spacing,
            _tolerance);
    }

    const std::size_t local_count = _local_positions_cache.size();
    if (local_count == 0) return;

    // Phase 2: Find available slots (active == 0)
    DeviceBuffer<int> slot_indices;
    const int available_count = build_available_slot_indices(
        particle_probe,
        slot_indices,
        local_count);
    if (available_count <= 0) return;

    const int emit_count = static_cast<int>(std::min<std::size_t>(
        local_count,
        static_cast<std::size_t>(available_count)));

    // Phase 3: Transform to world space and assign species
    DeviceBuffer<std::size_t> emitted_species;
    initialize_emitted_particles(
        particle_probe,
        slot_indices,
        emit_count,
        emitted_species);

    // Phase 4: Initialize velocities with generators
    assign_generated_velocities(
        particle_probe,
        slot_indices,
        emit_count,
        emitted_species);
}

template <typename T>
void
Source<T>::set_unit(Unit<T> unit) noexcept {
    _unit = std::move(unit);
}

template <typename T>
void
Source<T>::set_spawn_type(const SpawnType spawn_type) noexcept {
    _spawn_operator.type = spawn_type;
}

template <typename T>
void
Source<T>::set_spacing(const T spacing) noexcept {
    _spacing = spacing;
}

template <typename T>
void
Source<T>::set_tolerance(const T tolerance) noexcept {
    _tolerance = tolerance;
}

template <typename T>
void
Source<T>::set_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    update_species_distribution();
}

template <typename T>
const Unit<T>&
Source<T>::unit() const noexcept {
    return _unit;
}

template <typename T>
SpawnType
Source<T>::spawn_type() const noexcept {
    return _spawn_operator.type;
}

template <typename T>
T
Source<T>::spacing() const noexcept {
    return _spacing;
}

template <typename T>
T
Source<T>::tolerance() const noexcept {
    return _tolerance;
}

template <typename T>
const FluidHostPtr<T>&
Source<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
int
Source<T>::build_available_slot_indices(const ParticleDeviceProbe<T>& particle_probe,
                                        DeviceBuffer<int>& slot_indices,
                                        const std::size_t requested_count) const {
    slot_indices.clear();
    if (requested_count == 0 || particle_probe.particle_count <= 0) return 0;

    const int capacity = particle_probe.particle_count;
    int* active_ptr    = particle_probe.acitve;

    // If no active buffer, use first N slots
    if (active_ptr == nullptr) {
        const int selected_count = static_cast<int>(std::min<std::size_t>(
            requested_count,
            static_cast<std::size_t>(capacity)));
        if (selected_count <= 0) return 0;

        slot_indices.resize(static_cast<std::size_t>(selected_count), 0);
        int* slot_indices_ptr = atlas::raw_pointer_cast(slot_indices.data());
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            selected_count,
            [=] ATLAS_ALL_DEVICE(const int i) {
                slot_indices_ptr[i] = i;
            });
        return selected_count;
    }

    // Build mask of free slots (active == 0)
    DeviceBuffer<int> free_mask(static_cast<std::size_t>(capacity), 0);
    DeviceBuffer<int> free_offsets(static_cast<std::size_t>(capacity), 0);
    DeviceBuffer<int> free_indices(static_cast<std::size_t>(capacity), -1);

    int* free_mask_ptr    = atlas::raw_pointer_cast(free_mask.data());
    int* free_offsets_ptr = atlas::raw_pointer_cast(free_offsets.data());
    int* free_indices_ptr = atlas::raw_pointer_cast(free_indices.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        capacity,
        [=] ATLAS_ALL_DEVICE(const int i) {
            free_mask_ptr[i] = (active_ptr[i] == 0) ? 1 : 0;
        });

    atlas::exclusive_scan<ExecutionPolicy::device>(
        free_mask.begin(),
        free_mask.end(),
        free_offsets.begin(),
        0);

    // Count free slots
    int free_count = 0;
    {
        int last_mask = 0;
        int last_offset = 0;
        atlas::copy_device_to_host(
            atlas::raw_pointer_cast(free_mask.data()) + capacity - 1,
            &last_mask,
            1);
        atlas::copy_device_to_host(
            atlas::raw_pointer_cast(free_offsets.data()) + capacity - 1,
            &last_offset,
            1);
        free_count = last_offset + last_mask;
    }
    if (free_count <= 0) return 0;

    // Compact free indices
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        capacity,
        [=] ATLAS_ALL_DEVICE(const int i) {
            if (free_mask_ptr[i] == 0) return;
            free_indices_ptr[free_offsets_ptr[i]] = i;
        });

    const int selected_count = static_cast<int>(std::min<std::size_t>(
        requested_count,
        static_cast<std::size_t>(free_count)));
    if (selected_count <= 0) return 0;

    slot_indices.resize(static_cast<std::size_t>(selected_count), -1);
    int* slot_indices_ptr = atlas::raw_pointer_cast(slot_indices.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        selected_count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            slot_indices_ptr[i] = free_indices_ptr[i];
        });

    return selected_count;
}

template <typename T>
void
Source<T>::update_species_distribution() {
    _species_cdf_device.clear();
    _species_count = 0;

    if (_fluid && !_fluid->amounts().empty()) {
        const auto& amounts = _fluid->amounts();
        std::vector<T> cdf_host(amounts.size(), T(0));

        T cumulative = T(0);
        for (std::size_t i = 0; i < amounts.size(); ++i) {
            cumulative += amounts[i];
            cdf_host[i] = cumulative;
        }

        if (!cdf_host.empty() && cumulative > T(0)) {
            // Normalize to [0, 1]
            for (auto& val : cdf_host) {
                val /= cumulative;
            }
            cdf_host.back() = T(1);

            _species_cdf_device.resize(cdf_host.size());
            atlas::copy_host_to_device(
                cdf_host.data(),
                atlas::raw_pointer_cast(_species_cdf_device.data()),
                cdf_host.size());
            _species_count = static_cast<int>(cdf_host.size());
        }
    }
}

template <typename T>
void
Source<T>::initialize_emitted_particles(const ParticleDeviceProbe<T>& particle_probe,
                                        const DeviceBuffer<int>& slot_indices,
                                        const int emit_count,
                                        DeviceBuffer<std::size_t>& emitted_species) const {
    emitted_species.resize(static_cast<std::size_t>(emit_count), 0);

    const Vector3<T>* local_positions_ptr = atlas::raw_pointer_cast(_local_positions_cache.data());
    const int* slot_indices_ptr           = atlas::raw_pointer_cast(slot_indices.data());
    const T* cdf_device_ptr               = _species_cdf_device.empty() ? nullptr : atlas::raw_pointer_cast(_species_cdf_device.data());
    const int cdf_count                   = _species_count;
    const auto sync_op                    = _unit.sync_operator();
    Vector3<T>* pos_ptr                   = particle_probe.pos;
    Vector3<T>* vel_ptr                   = particle_probe.vel;
    int* active_ptr                       = particle_probe.acitve;
    std::size_t* species_ptr              = particle_probe.species;
    std::size_t* emitted_species_ptr      = atlas::raw_pointer_cast(emitted_species.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        emit_count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            const int slot = slot_indices_ptr[i];
            if (slot < 0) return;

            // Transform local position to world space
            pos_ptr[slot] = sync_op.sync_to_world(local_positions_ptr[i]);

            // Mark as active
            if (active_ptr != nullptr) active_ptr[slot] = 1;

            // Assign species using CDF
            std::size_t species = 0;
            if (cdf_device_ptr != nullptr && cdf_count > 0) {
                const T u = (static_cast<T>(i) + T(0.5)) / static_cast<T>(emit_count);
                while (species + 1 < static_cast<std::size_t>(cdf_count) && u > cdf_device_ptr[species]) {
                    ++species;
                }
            }

            emitted_species_ptr[i] = species;
            if (species_ptr != nullptr) species_ptr[slot] = species;

            // Initialize velocity to zero (will be set by generators)
            if (vel_ptr != nullptr) vel_ptr[slot] = Vector3<T>(T(0), T(0), T(0));
        });
}

template <typename T>
void
Source<T>::assign_generated_velocities(const ParticleDeviceProbe<T>& particle_probe,
                                       const DeviceBuffer<int>& slot_indices,
                                       const int emit_count,
                                       const DeviceBuffer<std::size_t>& emitted_species) const {
    if (particle_probe.vel == nullptr || !_fluid) return;

    const auto& generators = _fluid->generators();
    for (std::size_t species = 0; species < generators.size(); ++species) {
        assign_species_velocities(
            particle_probe,
            slot_indices,
            emit_count,
            emitted_species,
            species,
            generators[species]);
    }
}

template <typename T>
void
Source<T>::assign_species_velocities(const ParticleDeviceProbe<T>& particle_probe,
                                     const DeviceBuffer<int>& slot_indices,
                                     const int emit_count,
                                     const DeviceBuffer<std::size_t>& emitted_species,
                                     const std::size_t species,
                                     const GeneratorHostPtr<T>& generator) const {
    if (!generator) return;

    // Build mask for this species
    DeviceBuffer<int> species_mask(static_cast<std::size_t>(emit_count), 0);
    DeviceBuffer<int> species_offsets(static_cast<std::size_t>(emit_count), 0);
    int* species_mask_ptr                  = atlas::raw_pointer_cast(species_mask.data());
    int* species_offsets_ptr               = atlas::raw_pointer_cast(species_offsets.data());
    const std::size_t* emitted_species_ptr = atlas::raw_pointer_cast(emitted_species.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        emit_count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            species_mask_ptr[i] = (emitted_species_ptr[i] == species) ? 1 : 0;
        });

    atlas::exclusive_scan<ExecutionPolicy::device>(
        species_mask.begin(),
        species_mask.end(),
        species_offsets.begin(),
        0);

    // Count species particles
    int species_count = 0;
    {
        int last_mask = 0;
        int last_offset = 0;
        atlas::copy_device_to_host(
            atlas::raw_pointer_cast(species_mask.data()) + emit_count - 1,
            &last_mask,
            1);
        atlas::copy_device_to_host(
            atlas::raw_pointer_cast(species_offsets.data()) + emit_count - 1,
            &last_offset,
            1);
        species_count = last_offset + last_mask;
    }
    if (species_count <= 0) return;

    // Compact species indices
    DeviceBuffer<int> species_indices(static_cast<std::size_t>(species_count), 0);
    int* species_indices_ptr = atlas::raw_pointer_cast(species_indices.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        emit_count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            if (species_mask_ptr[i] == 0) return;
            species_indices_ptr[species_offsets_ptr[i]] = i;
        });

    // Generate velocities for this species
    DeviceBuffer<Vector3<T>> species_velocities(
        static_cast<std::size_t>(species_count),
        Vector3<T>(T(0), T(0), T(0)));
    generator->generate(species_velocities);

    // Assign generated velocities
    Vector3<T>* vel_ptr                      = particle_probe.vel;
    const int* slot_indices_ptr              = atlas::raw_pointer_cast(slot_indices.data());
    const Vector3<T>* species_velocities_ptr = atlas::raw_pointer_cast(species_velocities.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        species_count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            const int emit_idx = species_indices_ptr[i];
            const int slot     = slot_indices_ptr[emit_idx];
            if (slot < 0) return;
            vel_ptr[slot] = species_velocities_ptr[i];
        });
}

template <typename T>
Source<T>
Source<T>::Builder::build() {
    validate();
    return Source<T>(std::move(*_unit), _spawn_type, _spacing, _tolerance);
}

template <typename T>
atlas::host_shared_ptr<Source<T>>
Source<T>::Builder::make_host_shared() {
    return atlas::make_host_shared<Source<T>>(build());
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_unit(const Unit<T>& unit) noexcept {
    _unit = unit;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_unit(Unit<T>&& unit) noexcept {
    _unit = std::move(unit);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_type(const SpawnType spawn_type) noexcept {
    _spawn_type = spawn_type;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spacing(const T spacing) noexcept {
    _spacing = spacing;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_tolerance(const T tolerance) noexcept {
    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_fluid(const FluidHostPtr<T>& fluid) noexcept {
    _fluid = fluid;
    return *this;
}

template <typename T>
void
Source<T>::Builder::validate() const {
    if (!_unit.has_value()) {
        atlas::logger::error()
            << "Source::Builder: unit must be provided.";
        throw std::runtime_error("Source::Builder: unit must be provided.");
    }

    if (!std::isfinite(_spacing) || _spacing <= T(0)) {
        atlas::logger::error()
            << "Source::Builder: spacing must be positive and finite.";
        throw std::runtime_error("Source::Builder: spacing must be positive and finite.");
    }

    if (!std::isfinite(_tolerance)) {
        atlas::logger::error()
            << "Source::Builder: tolerance must be finite.";
        throw std::runtime_error("Source::Builder: tolerance must be finite.");
    }
}

} // namespace atlas::system
