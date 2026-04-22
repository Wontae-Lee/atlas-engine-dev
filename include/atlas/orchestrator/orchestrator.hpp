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
    // Store the optional universe/fluid dependencies and the ordered solver list.
}

template <typename T>
typename Orchestrator<T>::Builder
Orchestrator<T>::builder() noexcept {

    // Return a default-initialized builder for fluent Orchestrator construction.
    return Builder {};
}

template <typename T>
void
Orchestrator<T>::search() {

    if (_searcher) {
        _searcher->build();
    }
}

template <typename T>
void
Orchestrator<T>::classify() {

    if (_codec) {
        _codec->update();
    }
}

template <typename T>
void
Orchestrator<T>::measure() {

    if (_measurer) {
        _measurer->measure();
    }
}

template <typename T>
void
Orchestrator<T>::solve(const T dt) {

    // If there are no solvers, there is nothing to execute.
    if (_solvers.empty()) {
        return;
    }

    // When no codec is configured, execute each non-null solver through the
    // plain solve(dt) entry point with no codec-side orchestration context.
    if (!_codec) {

        for (const auto& solver : _solvers) {

            if (!solver) continue;

            solver->solve(dt);
        }

        return;
    }

    // When a codec is configured, retrieve the solver-allocation object owned
    // by the codec and pass it to every solver together with the solver index.
    //
    // This allows each solver to access codec-managed orchestration resources
    // while still knowing its position in the ordered solver sequence.
    const auto* allocated_solver = &_codec->allocated_solver();
    const int solver_count       = static_cast<int>(_solvers.size());
    for (int i = 0; i < solver_count; i++) {
        if (!_solvers[i]) continue;

        // Execute the solver in codec-aware mode.
        //
        // Parameters:
        // - allocated_solver : codec-owned solver allocation/context
        // - i                : current solver index in execution order
        _solvers[i]->solve(allocated_solver, i, dt);
    }
}

template <typename T>
void
Orchestrator<T>::update(const T dt) {

    orchestrate(dt);
}

template <typename T>
void
Orchestrator<T>::apply_field_force(const T dt) {

    // Applying a cell-wise force requires the universe field state, the fluid
    // velocity/species states, and the searcher-generated cell-to-particle map.
    if (!_universe || !_fluid || !_searcher || !(dt != T(0))) {
        return;
    }

    auto* field_force_state = _universe->template state<atlas::universe::UniverseFieldForceState<T>>();
    auto* velocity_state    = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state     = _fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    if (field_force_state == nullptr || velocity_state == nullptr || species_state == nullptr) {
        return;
    }

    auto& field_force         = field_force_state->data();
    auto& velocity            = velocity_state->data();
    auto& species             = species_state->data();
    auto& particle_properties = _fluid->particle_properties();

    const int particle_count   = static_cast<int>(_fluid->particle_count());
    const int num_of_cells     = static_cast<int>(field_force.size());
    const int num_of_species   = static_cast<int>(particle_properties.size());
    const auto* indices_ptr    = _searcher->indices();
    const auto* cell_start_ptr = _searcher->cell_start();
    const auto* cell_end_ptr   = _searcher->cell_end();

    if (particle_count <= 0 || num_of_cells <= 0 || num_of_species <= 0 || indices_ptr == nullptr
        || cell_start_ptr == nullptr || cell_end_ptr == nullptr) {
        return;
    }

    auto* force_ptr         = atlas::raw_pointer_cast(field_force.data());
    auto* velocity_ptr      = atlas::raw_pointer_cast(velocity.data());
    const auto* species_ptr = atlas::raw_pointer_cast(species.data());
    const auto* props_ptr   = atlas::raw_pointer_cast(particle_properties.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int start = cell_start_ptr[cell];
            const int end   = cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Vector3<T> force = force_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= particle_count) {
                    continue;
                }

                const std::size_t species_index = species_ptr[particle_index];
                if (species_index >= static_cast<std::size_t>(num_of_species)) {
                    continue;
                }

                const T mass = props_ptr[species_index].mass;
                if (!(mass > T(0))) {
                    continue;
                }

                velocity_ptr[particle_index] += force * (dt / mass);
            }
        });
}

template <typename T>
void
Orchestrator<T>::apply_gravity(const T dt) {

    if (!_universe || !_fluid || !_searcher || !(dt != T(0))) {
        return;
    }

    auto* gravity_state  = _universe->template state<atlas::universe::UniverseGravityState<T>>();
    auto* velocity_state = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    if (gravity_state == nullptr || velocity_state == nullptr) {
        return;
    }

    auto& gravity  = gravity_state->data();
    auto& velocity = velocity_state->data();

    const int particle_count   = static_cast<int>(_fluid->particle_count());
    const int num_of_cells     = static_cast<int>(gravity.size());
    const auto* indices_ptr    = _searcher->indices();
    const auto* cell_start_ptr = _searcher->cell_start();
    const auto* cell_end_ptr   = _searcher->cell_end();

    if (particle_count <= 0 || num_of_cells <= 0 || indices_ptr == nullptr || cell_start_ptr == nullptr
        || cell_end_ptr == nullptr) {
        return;
    }

    const auto* gravity_ptr = atlas::raw_pointer_cast(gravity.data());
    auto* velocity_ptr      = atlas::raw_pointer_cast(velocity.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int start = cell_start_ptr[cell];
            const int end   = cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Vector3<T> cell_gravity = gravity_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= particle_count) {
                    continue;
                }

                velocity_ptr[particle_index] += cell_gravity * dt;
            }
        });
}

template <typename T>
void
Orchestrator<T>::orchestrate(const T dt) {

    if (_searcher) {
        _searcher->invalidate();
    }

    search();
    classify();
    measure();
    apply_gravity(dt);
    apply_field_force(dt);
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

    // Replace the current codec with the supplied one.
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

    // Append a solver to the execution sequence.
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

    // Return the currently configured codec.
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

    // Return the ordered solver sequence.
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

    // Store the codec to be used by the constructed orchestrator.
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

    // Append a solver to the builder's solver sequence.
    _solvers.push_back(std::move(solver));
    return *this;
}

template <typename T>
void
Orchestrator<T>::Builder::validate() const {

    // Every configured solver must be valid.
    for (const auto& solver : _solvers) {

        if (!solver) {
            throw std::runtime_error("Orchestrator::Builder: solver must not be null.");
        }
    }
}

template <typename T>
void
Orchestrator<T>::Builder::ensure_gravity_state() const {

    if (!_universe || !_gravity.has_value()) {
        return;
    }

    const std::size_t number_of_cells = _universe->number_of_cells();
    if (number_of_cells == 0) {
        return;
    }

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

    // Validate builder state before constructing the value object.
    validate();
    ensure_gravity_state();
    return Orchestrator<T>(_universe, _fluid, _searcher, _codec, _measurer, _solvers);
}

template <typename T>
atlas::host_shared_ptr<Orchestrator<T>>
Orchestrator<T>::Builder::make_host_shared() const {

    // Validate builder state before constructing the shared object.
    validate();
    ensure_gravity_state();
    return atlas::make_host_shared<Orchestrator<T>>(_universe, _fluid, _searcher, _codec, _measurer, _solvers);
}

} // namespace atlas::system
