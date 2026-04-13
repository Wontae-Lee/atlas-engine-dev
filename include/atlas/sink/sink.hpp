#pragma once

#include <atlas/iterator/zip_iterator.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/tuple/tuple.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Sink<T>::Sink(DeviceBuffer<Unit<T>> units,
              DeviceBuffer<DespawnType> despawn_types,
              DeviceBuffer<DespawnOperator<T>> despawn_operators,
              const bool flip,
              const T tolerance) noexcept
    : _units(std::move(units))
    , _despawn_types(std::move(despawn_types))
    , _despawn_operators(std::move(despawn_operators))
    , _flip(flip)
    , _tolerance(tolerance) {
    // Construct a sink from already-materialized device buffers and runtime parameters.
    //
    // Stored state:
    // - _units              : sink units defining despawn geometry and motion
    // - _despawn_types      : despawn mode metadata
    // - _despawn_operators  : geometric despawn predicates
    // - _flip               : invert despawn decision if true
    // - _tolerance          : geometric tolerance used during despawn tests
}

template <typename T>
typename Sink<T>::Builder
Sink<T>::builder() noexcept {
    // Return a fresh builder for staged Sink<T> construction.
    return Builder {};
}

template <typename T>
void
Sink<T>::update(const T dt) {
    // Advance all sink units before evaluating particle removal.
    //
    // This keeps moving / rotating sink geometry synchronized with simulation time.

    // Nothing to update if there are no units or dt is non-positive.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    auto* units = atlas::raw_pointer_cast(_units.data());
    // Raw pointer to device-side sink units.

    // Update every sink unit in parallel.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });
}
template <typename T>
void
Sink<T>::sink(FluidDeviceProbe<T>& particle_probe) {
    // Remove particles from the active particle prefix when they satisfy
    // at least one configured despawn rule.
    //
    // Conceptually, this function treats the particle probe as a structure-of-arrays
    // container whose active range is [0, particle_count). The function does not
    // allocate a new probe. Instead, it compacts surviving particles in-place by
    // applying remove_if over a zipped view of:
    // - position
    // - velocity
    // - temperature
    // - species
    //
    // High-level algorithm:
    // 1. verify that there are active particles and a usable sink configuration
    // 2. gather raw pointers and scalar configuration for device-side evaluation
    // 3. form a zipped iterator over the active particle records
    // 4. test each particle against all sink units
    // 5. remove particles that satisfy the despawn condition
    // 6. update particle_count to match the compacted result

    // Abort immediately if there is nothing meaningful to process.
    //
    // Cases:
    // - particle_count <= 0
    //   -> there are no active particles in the probe
    //
    // - empty()
    //   -> sink configuration is incomplete, meaning at least one of:
    //      * no sink units
    //      * no despawn types
    //      * no despawn operators
    //
    // In both situations, removal is skipped with no side effects.
    if (particle_probe.particle_count <= 0 || empty()) return;

    // Extract a raw pointer to the device-side sink unit array.
    //
    // Each unit provides:
    // - a geometry_operator() used to define the despawn geometry
    // - a sync_operator() used to map particle positions into that unit's local space
    const auto* units = atlas::raw_pointer_cast(_units.data());

    // Extract a raw pointer to the device-side despawn operator array.
    //
    // These operators define how a local-space particle position is tested
    // against the unit geometry.
    const auto* despawn_operators = atlas::raw_pointer_cast(_despawn_operators.data());

    // Cache the number of sink units.
    //
    // Every active particle will potentially be tested against all of these units
    // until one unit decides the particle should despawn.
    const int unit_count = static_cast<int>(_units.size());

    // Cache the number of despawn operators.
    //
    // Supported configuration layouts:
    // - exactly 1 operator:
    //   broadcast the same operator to every sink unit
    //
    // - exactly unit_count operators:
    //   use one operator per corresponding unit
    //
    // The inner predicate also contains a fallback to index 0 if needed.
    const int despawn_operator_count = static_cast<int>(_despawn_operators.size());

    // Copy the sink flip flag into a local scalar for lambda capture.
    //
    // Meaning:
    // - flip == false :
    //   despawn predicate result is used directly
    //
    // - flip == true :
    //   despawn predicate result is inverted
    //
    // This allows the sink to act either as:
    // - "remove particles that are inside/on geometry"
    // or
    // - "remove particles that are outside/not on geometry"
    // depending on the despawn operator semantics.
    const bool flip = _flip;

    // Copy geometric tolerance into a local scalar for device capture.
    //
    // This tolerance is forwarded into each despawn operator call and typically
    // controls boundary robustness for geometric classification.
    const T tol = _tolerance;

    // Build a zipped iterator to the first active particle record.
    //
    // The zip groups multiple SoA arrays into one logical particle record:
    // - particle_probe.pos
    // - particle_probe.vel
    // - particle_probe.temperature
    // - particle_probe.species
    //
    // This is essential because remove_if must compact all particle attributes
    // together so they remain aligned after removals.
    auto zip_begin = atlas::make_zip_iterator(
        atlas::make_tuple(
            particle_probe.pos,
            particle_probe.vel,
            particle_probe.temperature,
            particle_probe.species));

    // Build the zipped end iterator for the active particle prefix.
    //
    // Only the prefix [zip_begin, zip_end) is considered valid / active.
    // Any storage beyond particle_count is ignored.
    auto zip_end = zip_begin + particle_probe.particle_count;

    // Perform in-place compaction of the active particle range.
    //
    // remove_if semantics:
    // - elements for which the predicate returns true are removed
    // - surviving elements are moved toward the front
    // - the returned iterator marks the new logical end of the active range
    //
    // Important:
    // - this preserves alignment between position / velocity / temperature / species
    //   because they are traversed and compacted as one zipped record
    auto new_end = atlas::remove_if(
        atlas::device,
        zip_begin,
        zip_end,
        [=] ATLAS_DEVICE(const atlas::tuple<Vector3<T>, Vector3<T>, T, size_t>& t) {
            // Read the particle world-space position from the zipped tuple.
            //
            // Tuple layout is:
            // - get<0> : position
            // - get<1> : velocity
            // - get<2> : temperature
            // - get<3> : species
            //
            // Only position is needed for despawn classification.
            const Vector3<T>& p = atlas::get<0>(t);

            // Track whether any sink unit decides that this particle should despawn.
            //
            // Initial assumption:
            // - particle survives unless at least one sink unit says otherwise
            bool should_despawn = false;

            // Test the current particle against every configured sink unit.
            //
            // Loop behavior:
            // - evaluate units in order
            // - stop immediately once any unit triggers despawn
            for (int i = 0; i < unit_count; ++i) {
                // Access the i-th sink unit.
                const auto& unit = units[i];

                // Access the transform operator of the current unit.
                //
                // This is used to map particle positions from world space into the
                // local coordinate system in which the unit geometry is defined.
                const auto& sync_op = unit.sync_operator();

                // Access the current unit's geometry query operator.
                //
                // This operator is the local-space geometry against which the particle
                // will be tested by the despawn operator.
                const auto& geometry_op = unit.geometry_operator();

                // Select which despawn operator to use for this unit.
                //
                // Resolution rules:
                // - if there is only one despawn operator, broadcast it to all units
                // - otherwise, use the unit-matching operator at index i
                // - if i exceeds the operator array size for any reason, fall back to 0
                //
                // This matches the same broadcast-or-per-unit design used in other
                // system components such as Source and Collider.
                const int despawn_operator_index
                    = (despawn_operator_count == 1 || i >= despawn_operator_count) ? 0 : i;

                // Transform the particle position from world space into the local space
                // of the current sink unit.
                //
                // Why local space is required:
                // - geometry_op is defined in the unit's own coordinate frame
                // - despawn tests must therefore operate on local coordinates
                const Vector3<T> local_p = sync_op.sync_to_local(p);

                // Evaluate the geometric despawn predicate for this particle against
                // the current unit.
                //
                // If the predicate returns true:
                // - the particle is marked for removal
                // - there is no need to test additional units
                if (despawn_operators[despawn_operator_index].despawn(geometry_op, local_p, tol)) {
                    should_despawn = true;
                    break;
                }
            }

            // Convert "should_despawn" into the final remove_if predicate result.
            //
            // Standard mode:
            // - flip == false
            //   -> return should_despawn directly
            //   -> particles selected by sink logic are removed
            //
            // Inverted mode:
            // - flip == true
            //   -> return logical negation of should_despawn
            //   -> particles NOT selected by sink logic are removed instead
            //
            // Since remove_if removes records for which the predicate returns true,
            // this line completely defines the sink's final retention/removal policy.
            return flip ? !should_despawn : should_despawn;
        });

    // Update the probe's active particle count after compaction.
    //
    // new_end - zip_begin yields the number of surviving zipped particle records.
    // That count becomes the new active prefix length for all SoA arrays.
    particle_probe.particle_count = static_cast<int>(new_end - zip_begin);
}

template <typename T>
void
Sink<T>::set_units(DeviceBuffer<Unit<T>> units) noexcept {
    // Replace device-side sink units.
    _units = std::move(units);
}

template <typename T>
void
Sink<T>::set_units(const HostBuffer<Unit<T>>& units) {
    // Replace sink units from host-side storage.

    if (units.empty()) {
        atlas::logger::error()
            << "Sink: units must not be empty.";
        throw std::runtime_error("Sink: units must not be empty.");
    }

    _units = DeviceBuffer<Unit<T>>(units.begin(), units.end());
}

template <typename T>
void
Sink<T>::set_despawn_operators(DeviceBuffer<DespawnOperator<T>> despawn_operators) noexcept {
    // Replace device-side despawn operators.
    _despawn_operators = std::move(despawn_operators);
}

template <typename T>
void
Sink<T>::set_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators) {
    // Replace despawn operators from host-side storage.

    if (despawn_operators.empty()) {
        atlas::logger::error()
            << "Sink: despawn operators must not be empty.";
        throw std::runtime_error("Sink: despawn operators must not be empty.");
    }

    _despawn_operators = DeviceBuffer<DespawnOperator<T>>(
        despawn_operators.begin(),
        despawn_operators.end());
}

template <typename T>
void
Sink<T>::set_despawn_types(DeviceBuffer<DespawnType> despawn_types) noexcept {
    // Replace device-side despawn types.
    _despawn_types = std::move(despawn_types);
}

template <typename T>
void
Sink<T>::set_despawn_types(const HostBuffer<DespawnType>& despawn_types) {
    // Replace despawn type metadata from host-side storage.

    if (despawn_types.empty()) {
        atlas::logger::error()
            << "Sink: despawn types must not be empty.";
        throw std::runtime_error("Sink: despawn types must not be empty.");
    }

    _despawn_types = DeviceBuffer<DespawnType>(despawn_types.begin(), despawn_types.end());
}

template <typename T>
void
Sink<T>::set_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept {
    // Convenience overload for broadcast despawn operator configuration.
    set_despawn_operators(DeviceBuffer<DespawnOperator<T>>(1, despawn_operator));
}

template <typename T>
void
Sink<T>::set_tolerance(const T tolerance) noexcept {
    // Update geometric despawn tolerance.
    _tolerance = tolerance;
}

template <typename T>
void
Sink<T>::set_flip(const bool flip) noexcept {
    // Update despawn decision inversion flag.
    _flip = flip;
}

template <typename T>
DeviceBuffer<Unit<T>>&
Sink<T>::units() noexcept {
    // Mutable access to sink units.
    return _units;
}

template <typename T>
const DeviceBuffer<Unit<T>>&
Sink<T>::units() const noexcept {
    // Const access to sink units.
    return _units;
}

template <typename T>
DeviceBuffer<DespawnOperator<T>>&
Sink<T>::despawn_operators() noexcept {
    // Mutable access to despawn operators.
    return _despawn_operators;
}

template <typename T>
const DeviceBuffer<DespawnOperator<T>>&
Sink<T>::despawn_operators() const noexcept {
    // Const access to despawn operators.
    return _despawn_operators;
}

template <typename T>
DeviceBuffer<DespawnType>&
Sink<T>::despawn_types() noexcept {
    // Mutable access to despawn type metadata.
    return _despawn_types;
}

template <typename T>
const DeviceBuffer<DespawnType>&
Sink<T>::despawn_types() const noexcept {
    // Const access to despawn type metadata.
    return _despawn_types;
}

template <typename T>
T
Sink<T>::tolerance() const noexcept {
    // Return current geometric despawn tolerance.
    return _tolerance;
}

template <typename T>
bool
Sink<T>::flip() const noexcept {
    // Return whether despawn decision is inverted.
    return _flip;
}

template <typename T>
bool
Sink<T>::empty() const noexcept {
    // Sink is considered unusable if any required configuration buffer is empty.
    return _units.empty() || _despawn_types.empty() || _despawn_operators.empty();
}

template <typename T>
Sink<T>
Sink<T>::Builder::build() {
    // Validate builder state before materializing the final sink.
    validate();

    auto despawn_types     = _despawn_types;
    auto despawn_operators = _despawn_operators;
    // Copy builder-side host buffers into local temporaries before conversion.

    // Materialize backend-native buffers in one place and construct the sink by value.
    return Sink<T>(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<DespawnType>(despawn_types.begin(), despawn_types.end()),
        DeviceBuffer<DespawnOperator<T>>(despawn_operators.begin(), despawn_operators.end()),
        _flip,
        _tolerance);
}

template <typename T>
atlas::host_shared_ptr<Sink<T>>
Sink<T>::Builder::make_host_shared() {
    // Build sink by value, then move into host-shared storage.
    return atlas::make_host_shared<Sink<T>>(build());
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
    // Append sink units to the builder.

    if (units.empty()) {
        atlas::logger::error()
            << "Sink::Builder: units must not be empty.";
        throw std::runtime_error("Sink::Builder: units must not be empty.");
    }

    _units.insert(_units.end(), units.begin(), units.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_types(const HostBuffer<DespawnType>& despawn_types) {
    // Append despawn type metadata to the builder.

    if (despawn_types.empty()) {
        atlas::logger::error()
            << "Sink::Builder: despawn types must not be empty.";
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }

    _despawn_types.insert(_despawn_types.end(), despawn_types.begin(), despawn_types.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept {
    // Append one despawn operator to the builder.
    _despawn_operators.push_back(despawn_operator);
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators) {
    // Append multiple despawn operators to the builder.

    if (despawn_operators.empty()) {
        atlas::logger::error()
            << "Sink::Builder: despawn operators must not be empty.";
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }

    _despawn_operators.insert(
        _despawn_operators.end(),
        despawn_operators.begin(),
        despawn_operators.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_tolerance(const T tolerance) noexcept {
    // Stage geometric despawn tolerance.
    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_flip(const bool flip) noexcept {
    // Stage despawn decision inversion flag.
    _flip = flip;
    return *this;
}

template <typename T>
void
Sink<T>::Builder::validate() const {
    // Validate all required builder-side sink configuration.

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

    if (!_despawn_operators.empty()
        && _despawn_operators.size() != 1
        && _despawn_operators.size() != _units.size()) {
        // Operators must either:
        // - broadcast with exactly one operator, or
        // - provide one operator per unit
        atlas::logger::error()
            << "Sink::Builder: despawn operators must have size 1 or match the unit count.";
        throw std::runtime_error(
            "Sink::Builder: despawn operators must have size 1 or match the unit count.");
    }

    if (!_despawn_types.empty()
        && _despawn_types.size() != 1
        && _despawn_types.size() != _units.size()) {
        // Despawn type metadata must either:
        // - broadcast with exactly one entry, or
        // - provide one entry per unit
        atlas::logger::error()
            << "Sink::Builder: despawn types must have size 1 or match the unit count.";
        throw std::runtime_error(
            "Sink::Builder: despawn types must have size 1 or match the unit count.");
    }

    if (!std::isfinite(_tolerance)) {
        // Tolerance must be finite so geometric despawn tests remain well-defined.
        atlas::logger::error()
            << "Sink::Builder: tolerance must be finite.";
        throw std::runtime_error("Sink::Builder: tolerance must be finite.");
    }
}

} // namespace atlas::system