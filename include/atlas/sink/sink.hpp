#pragma once

#include <atlas/iterator/zip_iterator.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/tuple/tuple.h>

#include <ranges>
#include <stdexcept>
#include <utility>

namespace atlas::fluid {

template <typename T>
Sink<T>::Sink(DeviceBuffer<Unit<T>> units,
              DeviceBuffer<DespawnType> despawn_types,
              DeviceBuffer<DespawnOperator<T>> despawn_operators,
              atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
              const bool flip,
              const T tolerance) noexcept
    : _units(std::move(units))
    , _despawn_types(std::move(despawn_types))
    , _despawn_operators(std::move(despawn_operators))
    , _fluid(std::move(fluid))
    , _flip(flip)
    , _tolerance(tolerance) {

    // Store the fully prepared sink configuration.
    //
    // At this point:
    // - _units holds the sink geometry and transforms on the device,
    // - _despawn_types holds the configured runtime type tags,
    // - _despawn_operators holds the actual despawn decision logic,
    // - _fluid points to the target fluid,
    // - _flip and _tolerance define the sink evaluation policy.
}

template <typename T>
typename Sink<T>::Builder
Sink<T>::builder() noexcept {

    // Return a default-initialized builder for fluent Sink construction.
    return Builder {};
}

template <typename T>
void
Sink<T>::update(const T dt) {

    // A meaningful update requires:
    // - at least one sink unit,
    // - and a strictly positive time step.
    //
    // If either condition fails, skip both unit animation/update and sink processing.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    auto* units = atlas::raw_pointer_cast(_units.data());

    // Advance every sink unit independently on the device.
    //
    // Units may carry time-dependent transforms or motion through Unit::update(dt),
    // which means the sink geometry can move or rotate over time.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });

    // After unit transforms have been updated, apply sink logic to the fluid.
    sink();
}

template <typename T>
void
Sink<T>::sink() {

    // Sink processing requires:
    // - a valid fluid object,
    // - a position state, because particle locations are tested against geometry,
    // - an active state, because particles are marked alive/dead through that mask.
    if (!_fluid) {
        return;
    }

    auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* active_state   = _fluid->template state<atlas::fluid::FluidActiveState<T>>();

    if (position_state == nullptr || active_state == nullptr) {
        return;
    }

    auto& positions = position_state->data();
    auto& active    = active_state->data();

    // If the underlying buffers are empty, there is nothing to test or remove.
    if (positions.empty() || active.empty()) {
        return;
    }

    const auto* units             = atlas::raw_pointer_cast(_units.data());
    const auto* despawn_operators = atlas::raw_pointer_cast(_despawn_operators.data());
    const auto* positions_ptr     = atlas::raw_pointer_cast(positions.data());
    auto* active_ptr              = atlas::raw_pointer_cast(active.data());

    const int unit_count             = static_cast<int>(_units.size());
    const int despawn_operator_count = static_cast<int>(_despawn_operators.size());
    const std::size_t particle_count = _fluid->particle_count();
    const bool flip                  = _flip;
    const T tol                      = _tolerance;

    // Process each particle independently on the device.
    //
    // For each active particle:
    // 1. transform the particle into each unit's local space,
    // 2. evaluate the configured despawn rule against that unit,
    // 3. stop at the first matching unit,
    // 4. convert the result into keep/remove semantics,
    // 5. write the updated active flag.
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        particle_count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            // Ignore particles that are already inactive.
            if (active_ptr[i] == 0) {
                return;
            }

            const Vector3<T>& p = positions_ptr[i];
            bool should_despawn = false;

            // Test the particle against every configured sink unit.
            for (int unit_index = 0; unit_index < unit_count; ++unit_index) {
                const auto& unit        = units[unit_index];
                const auto& sync_op     = unit.sync_operator();
                const auto& geometry_op = unit.geometry_operator();

                // Despawn operators may be configured in two supported layouts:
                // - exactly one shared operator for all units,
                // - one operator per unit.
                //
                // If only one operator exists, use index 0 for every unit.
                const int despawn_operator_index
                    = (despawn_operator_count == 1 || unit_index >= despawn_operator_count) ? 0 : unit_index;

                // Convert the particle position from world space into the local
                // space of the current sink unit before querying its geometry.
                const Vector3<T> local_p = sync_op.sync_to_local(p);

                // Stop at the first unit that requests particle removal.
                if (despawn_operators[despawn_operator_index].despawn(geometry_op, local_p, tol)) {
                    should_despawn = true;
                    break;
                }
            }

            // Standard behavior:
            // - despawn match => remove particle
            // - no match      => keep particle
            //
            // Flipped behavior inverts that interpretation.
            const bool keep_particle = flip ? should_despawn : !should_despawn;

            // Encode the survival result back into the active-state mask.
            active_ptr[i] = keep_particle ? 1 : 0;
        });

    // Re-pack every registered fluid state so surviving particles occupy a dense prefix.
    compact_fluid_particles();
}

template <typename T>
void
Sink<T>::compact_fluid_particles() {

    // Compaction is driven by the fluid's active-state mask.
    auto* active_state = _fluid->template state<atlas::fluid::FluidActiveState<T>>();
    if (active_state == nullptr) {
        return;
    }

    auto& active     = active_state->data();
    const auto count = _fluid->particle_count();

    // No active prefix means there is nothing to compact.
    if (count == 0) {
        return;
    }

    // Ensure all scratch buffers are large enough for the current active prefix.
    //
    // _keep[i]            : 1 if particle i survives, otherwise 0
    // _offsets[i]         : exclusive prefix sum of _keep
    // _compact_indices[k] : source particle index that moves into compacted slot k
    if (_keep.size() != count) {
        _keep.resize(count);
    }

    if (_offsets.size() != count) {
        _offsets.resize(count);
    }

    if (_compact_indices.size() != count) {
        _compact_indices.resize(count);
    }

    const auto* active_ptr = atlas::raw_pointer_cast(active.data());
    auto* keep_ptr         = atlas::raw_pointer_cast(_keep.data());

    // Convert active flags into a binary keep mask.
    //
    // Convention:
    // - active != 0 => keep = 1
    // - active == 0 => keep = 0
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            keep_ptr[i] = active_ptr[i] == 0 ? std::size_t { 0 } : std::size_t { 1 };
        });

    // Compute exclusive prefix offsets over the keep mask.
    //
    // Example:
    //   keep    = [1, 0, 1, 1, 0]
    //   offsets = [0, 1, 1, 2, 3]
    //
    // For a surviving particle i, offsets[i] is its destination index
    // in the compacted dense prefix.
    atlas::exclusive_scan<ExecutionPolicy::device>(
        _keep.begin(),
        _keep.begin() + static_cast<std::ptrdiff_t>(count),
        _offsets.begin(),
        std::size_t { 0 });

    // The final survivor count equals:
    //   last_offset + last_keep
    const std::size_t kept = _offsets[count - 1] + _keep[count - 1];

    // Fast path: if every particle survives, only update the logical count.
    if (kept == count) {
        _fluid->set_particle_count(kept);
        return;
    }

    // Fast path: if every particle is removed, clear the active prefix and
    // update the logical particle count to zero.
    if (kept == 0) {
        atlas::parallel_fill<ExecutionPolicy::device>(
            active.begin(),
            active.begin() + static_cast<std::ptrdiff_t>(count),
            0);
        _fluid->set_particle_count(kept);
        return;
    }

    const auto* offsets_ptr = atlas::raw_pointer_cast(_offsets.data());
    auto* indices_ptr       = atlas::raw_pointer_cast(_compact_indices.data());

    // Build the compacted destination-to-source mapping.
    //
    // For every surviving source index i:
    //   destination = offsets[i]
    //   compact_indices[destination] = i
    //
    // After this pass, compact_indices[k] tells every state which original
    // particle should become the k-th particle in the compacted layout.
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            if (keep_ptr[i] != 0) {
                indices_ptr[offsets_ptr[i]] = i;
            }
        });

    // Compact every registered fluid state with the exact same index map.
    //
    // This is critical for keeping all per-particle states aligned after removal.
    //
    // Typical examples:
    // - positions
    // - velocities
    // - species identifiers
    // - active flags
    // - any custom user-installed state
    for (auto& state : _fluid->states() | std::views::values) {
        if (state) {
            state->compact(_compact_indices, kept);
        }
    }

    // Clear the inactive tail beyond the compacted dense prefix inside the
    // active-state buffer so the full fluid storage remains consistent.
    atlas::parallel_fill<ExecutionPolicy::device>(
        active.begin() + static_cast<std::ptrdiff_t>(kept),
        active.begin() + static_cast<std::ptrdiff_t>(_fluid->buffer_size()),
        0);

    // Publish the new logical particle count.
    _fluid->set_particle_count(kept);
}

template <typename T>
Sink<T>
Sink<T>::Builder::build() {

    // Validate all structural and numeric builder parameters first.
    validate();

    auto despawn_types     = _despawn_types;
    auto despawn_operators = _despawn_operators;

    // Materialize the final sink using device-backed copies of the collected
    // host-side configuration buffers.
    return Sink<T>(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<DespawnType>(despawn_types.begin(), despawn_types.end()),
        DeviceBuffer<DespawnOperator<T>>(despawn_operators.begin(), despawn_operators.end()),
        _fluid,
        _flip,
        _tolerance);
}

template <typename T>
atlas::host_shared_ptr<Sink<T>>
Sink<T>::Builder::make_host_shared() {

    // Build a value object first, then move it into shared host ownership.
    return atlas::make_host_shared<Sink<T>>(build());
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {

    if (units.empty()) {
        throw std::runtime_error("Sink::Builder: units must not be empty.");
    }

    // Append sink units instead of replacing the existing set.
    //
    // This allows callers to build a sink incrementally across multiple calls.
    _units.insert(_units.end(), units.begin(), units.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {

    // Store the target fluid whose particles will be tested and compacted by the sink.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_types(const HostBuffer<DespawnType>& despawn_types) {

    if (despawn_types.empty()) {
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }

    // Append despawn-type entries rather than replacing them.
    _despawn_types.insert(_despawn_types.end(), despawn_types.begin(), despawn_types.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept {

    // Append a single despawn operator entry.
    _despawn_operators.push_back(despawn_operator);
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators) {

    if (despawn_operators.empty()) {
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }

    // Append multiple despawn operators rather than replacing the existing set.
    _despawn_operators.insert(
        _despawn_operators.end(),
        despawn_operators.begin(),
        despawn_operators.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_tolerance(const T tolerance) noexcept {

    // Store the geometric tolerance used during despawn evaluation.
    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_flip(const bool flip) noexcept {

    // Store whether keep/remove semantics should be inverted.
    _flip = flip;
    return *this;
}

template <typename T>
void
Sink<T>::Builder::validate() const {

    // A sink always requires a valid target fluid.
    if (!_fluid) {
        throw std::runtime_error("Sink::Builder: fluid must not be null.");
    }

    // At least one sink unit must exist, otherwise there is no sink geometry.
    if (_units.empty()) {
        throw std::runtime_error("Sink::Builder: at least one unit must be provided.");
    }

    // Despawn type configuration must be present.
    if (_despawn_types.empty()) {
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }

    // Despawn operator configuration must also be present.
    if (_despawn_operators.empty()) {
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }

    // Supported despawn-operator layouts:
    // - size == 1              : shared by all units
    // - size == number of units: one per unit
    if (_despawn_operators.size() != 1
        && _despawn_operators.size() != _units.size()) {
        throw std::runtime_error(
            "Sink::Builder: despawn operators must have size 1 or match the unit count.");
    }

    // Supported despawn-type layouts follow the same cardinality rule.
    if (_despawn_types.size() != 1
        && _despawn_types.size() != _units.size()) {
        throw std::runtime_error(
            "Sink::Builder: despawn types must have size 1 or match the unit count.");
    }

    // Tolerance must be a finite numeric value.
    if (!std::isfinite(_tolerance)) {
        throw std::runtime_error("Sink::Builder: tolerance must be finite.");
    }
}

} // namespace atlas::fluid