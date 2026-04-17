#pragma once

#include <atlas/iterator/zip_iterator.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/tuple/tuple.h>

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
    // Store sink configuration and the target fluid.
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

    // No update is needed when there are no units or the time step is invalid.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    auto* units = atlas::raw_pointer_cast(_units.data());

    // Advance every sink unit independently on the device.
    //
    // A unit may internally update its transform, animation state,
    // or other time-dependent sink properties.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });
}

template <typename T>
void
Sink<T>::sink() {

    // Sink processing requires position and active-state buffers because:
    // - particle positions are tested against sink geometry
    // - active flags are updated to mark particles for removal
    auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* active_state   = _fluid->template state<atlas::fluid::FluidActiveState<T>>();

    if (position_state == nullptr || active_state == nullptr) {
        return;
    }

    auto& positions = position_state->data();
    auto& active    = active_state->data();

    // If there is no particle storage, there is nothing to test or remove.
    if (positions.empty() || active.empty()) {
        return;
    }

    const auto* units                = atlas::raw_pointer_cast(_units.data());
    const auto* despawn_operators    = atlas::raw_pointer_cast(_despawn_operators.data());
    const auto* positions_ptr        = atlas::raw_pointer_cast(positions.data());
    auto* active_ptr                 = atlas::raw_pointer_cast(active.data());
    const int unit_count             = static_cast<int>(_units.size());
    const int despawn_operator_count = static_cast<int>(_despawn_operators.size());
    const std::size_t particle_count = _fluid->particle_count();
    const bool flip                  = _flip;
    const T tol                      = _tolerance;

    // Process each particle independently on the device.
    //
    // For each active particle:
    // 1. test it against every sink unit
    // 2. decide whether it should be despawned
    // 3. update its active flag accordingly
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        particle_count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            // Skip already inactive particles.
            if (active_ptr[i] == 0) {
                return;
            }

            const Vector3<T>& p = positions_ptr[i];
            bool should_despawn = false;

            for (int unit_index = 0; unit_index < unit_count; ++unit_index) {
                const auto& unit        = units[unit_index];
                const auto& sync_op     = unit.sync_operator();
                const auto& geometry_op = unit.geometry_operator();

                // Despawn operators may be configured either:
                // - as a single shared operator for all units, or
                // - one operator per unit
                const int despawn_operator_index
                    = (despawn_operator_count == 1 || unit_index >= despawn_operator_count) ? 0 : unit_index;

                // Convert the world-space particle position into the unit's local
                // space before evaluating the unit geometry.
                const Vector3<T> local_p = sync_op.sync_to_local(p);

                if (despawn_operators[despawn_operator_index].despawn(geometry_op, local_p, tol)) {
                    should_despawn = true;
                    break;
                }
            }

            // Standard behavior:
            // - despawn match => remove particle
            // - no match      => keep particle
            //
            // When flip is enabled, that interpretation is inverted.
            const bool keep_particle = flip ? should_despawn : !should_despawn;
            active_ptr[i]            = keep_particle ? 1 : 0;
        });

    // Compact particle storage so inactive particles are physically removed from
    // the active prefix of the fluid buffers.
    _fluid->remove_particles();
}

template <typename T>
Sink<T>
Sink<T>::Builder::build() {

    // Validate all structural and numeric builder parameters before constructing
    // the final sink object.
    validate();

    auto despawn_types     = _despawn_types;
    auto despawn_operators = _despawn_operators;

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

    // Construct a value object first, then move it into shared host ownership.
    return atlas::make_host_shared<Sink<T>>(build());
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {

    if (units.empty()) {
        atlas::logger::error()
            << "Sink::Builder: units must not be empty.";
        throw std::runtime_error("Sink::Builder: units must not be empty.");
    }

    // Append sink units rather than replacing them, allowing incremental builder configuration.
    _units.insert(_units.end(), units.begin(), units.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {

    // Store the target fluid whose particles will be removed by this sink.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_types(const HostBuffer<DespawnType>& despawn_types) {

    if (despawn_types.empty()) {
        atlas::logger::error()
            << "Sink::Builder: despawn types must not be empty.";
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }

    // Append despawn type configuration entries.
    _despawn_types.insert(_despawn_types.end(), despawn_types.begin(), despawn_types.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept {

    // Add a single despawn operator entry.
    _despawn_operators.push_back(despawn_operator);
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators) {

    if (despawn_operators.empty()) {
        atlas::logger::error()
            << "Sink::Builder: despawn operators must not be empty.";
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }

    // Append despawn operators rather than replacing them.
    _despawn_operators.insert(
        _despawn_operators.end(),
        despawn_operators.begin(),
        despawn_operators.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_tolerance(const T tolerance) noexcept {

    // Store the geometric tolerance used by despawn checks.
    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_flip(const bool flip) noexcept {

    // Invert despawn acceptance when enabled.
    _flip = flip;
    return *this;
}

template <typename T>
void
Sink<T>::Builder::validate() const {

    if (!_fluid) {
        atlas::logger::error()
            << "Sink::Builder: fluid must not be null.";
        throw std::runtime_error("Sink::Builder: fluid must not be null.");
    }

    if (_units.empty()) {
        atlas::logger::error()
            << "Sink::Builder: at least one unit must be provided.";
        throw std::runtime_error("Sink::Builder: at least one unit must be provided.");
    }

    if (_despawn_types.empty()) {
        atlas::logger::error()
            << "Sink::Builder: despawn types must not be empty.";
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }

    if (_despawn_operators.empty()) {
        atlas::logger::error()
            << "Sink::Builder: despawn operators must not be empty.";
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }

    // Despawn operators must be either:
    // - shared by all units with exactly one entry, or
    // - specified per unit with matching unit count.
    if (!_despawn_operators.empty()
        && _despawn_operators.size() != 1
        && _despawn_operators.size() != _units.size()) {

        atlas::logger::error()
            << "Sink::Builder: despawn operators must have size 1 or match the unit count.";
        throw std::runtime_error(
            "Sink::Builder: despawn operators must have size 1 or match the unit count.");
    }

    // The same cardinality rule applies to despawn types.
    if (!_despawn_types.empty()
        && _despawn_types.size() != 1
        && _despawn_types.size() != _units.size()) {

        atlas::logger::error()
            << "Sink::Builder: despawn types must have size 1 or match the unit count.";
        throw std::runtime_error(
            "Sink::Builder: despawn types must have size 1 or match the unit count.");
    }

    if (!std::isfinite(_tolerance)) {

        atlas::logger::error()
            << "Sink::Builder: tolerance must be finite.";
        throw std::runtime_error("Sink::Builder: tolerance must be finite.");
    }
}

} // namespace atlas::fluid