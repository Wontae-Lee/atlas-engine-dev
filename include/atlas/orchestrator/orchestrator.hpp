#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/universe/universe_state.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
Orchestrator<T>::Orchestrator(UniverseHostPtr<T> universe,
                              FluidHostPtr<T> fluid,
                              SpatialHashingSearcherHostPtr<T> searcher,
                              CodecHostPtr<T> codec,
                              MeasurerHostPtr<T> measurer,
                              HostBuffer<SolveHostPtr<T>> solvers) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher))
    , _codec(std::move(codec))
    , _measurer(std::move(measurer))
    , _solvers(std::move(solvers)) {
}

template <typename T>
typename Orchestrator<T>::Builder
Orchestrator<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Orchestrator<T>::search() {
    // Rebuild the spatial search structure when a searcher is available.
    if (_searcher) {
        _searcher->build();
    }
}

template <typename T>
void
Orchestrator<T>::classify() {
    // Update solver allocation or particle classification when a codec is available.
    if (_codec) {
        _codec->update();
    }
}

template <typename T>
void
Orchestrator<T>::measure() {
    // Evaluate measurement states before force application and solver execution.
    if (_measurer) {
        _measurer->measure();
    }
}

template <typename T>
void
Orchestrator<T>::solve(const T dt) {
    // Nothing can be solved without registered solvers.
    if (_solvers.empty()) {
        return;
    }

    // If no codec is available, run every solver on the full particle set.
    if (!_codec) {
        for (const auto& solver : _solvers) {
            if (!solver) continue;

            solver->solve(dt);
        }

        return;
    }

    // When a codec exists, each solver receives the solver-allocation map.
    const auto* allocated_solver = &_codec->allocated_solver();
    const int solver_count       = static_cast<int>(_solvers.size());

    for (int i = 0; i < solver_count; i++) {
        if (!_solvers[i]) continue;

        _solvers[i]->solve(allocated_solver, i, dt);
    }
}

template <typename T>
void
Orchestrator<T>::update(const T dt) {
    // Forward the public update call to the complete orchestration sequence.
    orchestrate(dt);
}

template <typename T>
void
Orchestrator<T>::apply_field_force(const OrchestratorProbe& probe, const T dt) {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.field_force_cell_count,
        [=] ATLAS_DEVICE(const int cell) {
            // Read the sorted particle range belonging to this cell.
            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            // Field force is stored per cell and applied to every particle in that cell.
            const Vector3<T> force = probe.field_force_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                // Skip invalid particle indices defensively.
                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                const std::size_t species_index = probe.species_ptr[particle_index];

                // Skip particles whose species index is outside the property table.
                if (species_index >= static_cast<std::size_t>(probe.num_of_species)) {
                    continue;
                }

                const T mass = probe.properties_ptr[species_index].mass;

                // A non-positive mass cannot be used for acceleration computation.
                if (!(mass > T(0))) {
                    continue;
                }

                // Explicit velocity update: dv = (F / m) * dt.
                probe.velocity_ptr[particle_index] += force * (dt / mass);
            }
        });
}

template <typename T>
void
Orchestrator<T>::apply_gravity(const OrchestratorProbe& probe, const T dt) {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.gravity_cell_count,
        [=] ATLAS_DEVICE(const int cell) {
            // Read the sorted particle range belonging to this cell.
            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            // Gravity is stored per cell and applied as acceleration.
            const Vector3<T> cell_gravity = probe.gravity_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                // Skip invalid particle indices defensively.
                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                // Explicit velocity update: dv = g * dt.
                probe.velocity_ptr[particle_index] += cell_gravity * dt;
            }
        });
}

template <typename T>
bool
Orchestrator<T>::make_probe() noexcept {
    _probe = {};
    // The probe requires the core simulation objects and the spatial searcher.
    if (!_universe || !_fluid || !_searcher) {
        return false;
    }

    auto* field_force_state = _universe->template state<atlas::universe::UniverseFieldForceState<T>>();
    auto* gravity_state     = _universe->template state<atlas::universe::UniverseGravityState<T>>();
    auto* velocity_state    = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state     = _fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    // Velocity is mandatory because force and gravity both update particle velocity.
    if (velocity_state == nullptr) {
        return false;
    }

    auto& velocity = velocity_state->data();

    // Store common simulation sizes and searcher buffers.
    _probe.particle_count = static_cast<int>(_fluid->particle_count());
    _probe.num_of_cells   = _universe->number_of_cells();
    _probe.indices_ptr    = _searcher->indices();
    _probe.cell_start_ptr = _searcher->cell_start();
    _probe.cell_end_ptr   = _searcher->cell_end();
    _probe.velocity_ptr   = atlas::raw_pointer_cast(velocity.data());

    // Validate the minimum data required for cell-based particle traversal.
    if (velocity.empty() || _probe.particle_count <= 0 || _probe.num_of_cells <= 0
        || _probe.indices_ptr == nullptr || _probe.cell_start_ptr == nullptr
        || _probe.cell_end_ptr == nullptr) {
        return false;
    }

    // Species and particle properties are optional, but required for field-force mass lookup.
    if (species_state != nullptr) {
        auto& species             = species_state->data();
        auto& particle_properties = _fluid->particle_properties();

        if (!species.empty() && !particle_properties.empty()) {
            _probe.species_ptr    = atlas::raw_pointer_cast(species.data());
            _probe.properties_ptr = atlas::raw_pointer_cast(particle_properties.data());
            _probe.num_of_species = static_cast<int>(particle_properties.size());
        }
    }

    // Field force is optional and is attached only when the corresponding state exists.
    if (field_force_state != nullptr) {
        auto& field_force = field_force_state->data();

        if (!field_force.empty()) {
            _probe.field_force_ptr        = atlas::raw_pointer_cast(field_force.data());
            _probe.field_force_cell_count = static_cast<int>(field_force.size());
        }
    }

    // Gravity is optional and is attached only when the corresponding state exists.
    if (gravity_state != nullptr) {
        auto& gravity = gravity_state->data();

        if (!gravity.empty()) {
            _probe.gravity_ptr        = atlas::raw_pointer_cast(gravity.data());
            _probe.gravity_cell_count = static_cast<int>(gravity.size());
        }
    }

    return true;
}

template <typename T>
void
Orchestrator<T>::orchestrate(const T dt) {
    // Mark the previous spatial search result as stale before rebuilding it.
    if (_searcher) {
        _searcher->invalidate();
    }

    // Execute the simulation update pipeline in dependency order.
    search();
    classify();
    measure();

    if (dt != T(0)) {
        if (make_probe()) {
            const auto probe = _probe;
            if (probe.gravity_ptr != nullptr) {
                apply_gravity(probe, dt);
            }
            if (probe.field_force_ptr != nullptr && probe.species_ptr != nullptr
                && probe.properties_ptr != nullptr && probe.num_of_species > 0) {
                apply_field_force(probe, dt);
            }
        }
    }

    solve(dt);
}

template <typename T>
void
Orchestrator<T>::set_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
}

template <typename T>
void
Orchestrator<T>::set_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
}

template <typename T>
void
Orchestrator<T>::set_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
}

template <typename T>
void
Orchestrator<T>::set_codec(CodecHostPtr<T> codec) noexcept {
    _codec = std::move(codec);
}

template <typename T>
void
Orchestrator<T>::set_measurer(MeasurerHostPtr<T> measurer) noexcept {
    _measurer = std::move(measurer);
}

template <typename T>
void
Orchestrator<T>::add_solver(SolveHostPtr<T> solver) noexcept {
    _solvers.push_back(std::move(solver));
}

template <typename T>
const SpatialHashingSearcherHostPtr<T>&
Orchestrator<T>::searcher() const noexcept {
    return _searcher;
}

template <typename T>
const UniverseHostPtr<T>&
Orchestrator<T>::universe() const noexcept {
    return _universe;
}

template <typename T>
const FluidHostPtr<T>&
Orchestrator<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
const CodecHostPtr<T>&
Orchestrator<T>::codec() const noexcept {
    return _codec;
}

template <typename T>
const MeasurerHostPtr<T>&
Orchestrator<T>::measurer() const noexcept {
    return _measurer;
}

template <typename T>
const HostBuffer<SolveHostPtr<T>>&
Orchestrator<T>::solvers() const noexcept {
    return _solvers;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_codec(CodecHostPtr<T> codec) noexcept {
    _codec = std::move(codec);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_measurer(MeasurerHostPtr<T> measurer) noexcept {
    _measurer = std::move(measurer);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_gravity(const Vector3<T>& gravity) noexcept {
    _gravity = gravity;
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_solver(SolveHostPtr<T> solver) noexcept {
    _solvers.push_back(std::move(solver));
    return *this;
}

template <typename T>
void
Orchestrator<T>::Builder::validate() const {
    // Solvers are optional as a collection, but inserted solver entries must be valid.
    for (const auto& solver : _solvers) {
        if (!solver) {
            throw std::runtime_error("Orchestrator::Builder: solver must not be null.");
        }
    }
}

template <typename T>
void
Orchestrator<T>::Builder::ensure_gravity_state() const {
    // Gravity state is created only when both universe and gravity value are available.
    if (!_universe || !_gravity.has_value()) {
        return;
    }

    const std::size_t number_of_cells = _universe->number_of_cells();

    if (number_of_cells == 0) {
        return;
    }

    // If the gravity state already exists, resize it if necessary and refill it.
    if (_universe->template has_state<atlas::universe::UniverseGravityState<T>>()) {
        auto* gravity_state = _universe->template state<atlas::universe::UniverseGravityState<T>>();

        if (gravity_state != nullptr) {
            if (gravity_state->size() != number_of_cells) {
                gravity_state->data().resize(number_of_cells);
            }

            atlas::parallel_fill<ExecutionPolicy::device>(
                gravity_state->data().begin(),
                gravity_state->data().end(),
                *_gravity);
        }

        return;
    }

    // Otherwise, create a new gravity state and initialize every cell with the same value.
    _universe->template emplace_state<atlas::universe::UniverseGravityState<T>>(number_of_cells);

    auto* gravity_state = _universe->template state<atlas::universe::UniverseGravityState<T>>();

    if (gravity_state == nullptr) {
        return;
    }

    atlas::parallel_fill<ExecutionPolicy::device>(
        gravity_state->data().begin(),
        gravity_state->data().end(),
        *_gravity);
}

template <typename T>
Orchestrator<T>
Orchestrator<T>::Builder::build() const {
    validate();
    ensure_gravity_state();

    return Orchestrator<T>(_universe, _fluid, _searcher, _codec, _measurer, _solvers);
}

template <typename T>
atlas::host_shared_ptr<Orchestrator<T>>
Orchestrator<T>::Builder::make_host_shared() const {
    validate();
    ensure_gravity_state();

    return atlas::make_host_shared<Orchestrator<T>>(
        _universe,
        _fluid,
        _searcher,
        _codec,
        _measurer,
        _solvers);
}

} // namespace atlas::system
