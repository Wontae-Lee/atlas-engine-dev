#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
DsmcDisjointPairSolver<T>::DsmcDisjointPairSolver(UniverseHostPtr<T> universe,
                                                  FluidHostPtr<T> fluid,
                                                  SpatialHashingSearcherHostPtr<T> searcher,
                                                  const DsmcKernelType kernel_type) noexcept
    // Forward all core runtime dependencies to the common DSMC solver base class.
    //
    // This derived solver does not introduce a separate ownership model for the
    // universe, fluid, or searcher. Instead, it relies on the base DsmcSolver<T>
    // to store and manage those shared runtime handles together with the selected
    // collision kernel.
    //
    // Using std::move here avoids an unnecessary shared-pointer copy when the
    // constructor is given temporary or movable host handles.
    : DsmcSolver<T>(std::move(universe), std::move(fluid), std::move(searcher), kernel_type) { }

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder
DsmcDisjointPairSolver<T>::builder() noexcept {
    // Return a fresh builder with:
    // - null universe / fluid / searcher dependencies
    // - default hard-sphere kernel selection
    //
    // No validation happens here. Validation is deferred until build time so that
    // callers may fill the builder incrementally.
    return Builder {};
}

template <typename T>
void
DsmcDisjointPairSolver<T>::apply_collisions(const DeviceBuffer<int>* allocated_solver,
                                            const int index,
                                            const T) {
    // -------------------------------------------------------------------------
    // Step 1. Acquire every runtime state required by this collision strategy.
    // -------------------------------------------------------------------------
    //
    // This solver needs:
    // - particle velocities, because collision updates modify them in place
    // - particle species ids, because collision kernels need material properties
    // - per-cell particle counts, to know how many local pairs can be formed
    // - per-cell collision counts, to know how many collisions to attempt
    //
    // If any of these states are missing, this solver cannot safely perform its
    // work and therefore exits immediately.
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    // Abort early when the required simulation state layout is incomplete.
    //
    // This is intentionally a silent return instead of an exception because the
    // overall runtime may legally run different pipelines with different state
    // combinations depending on how the user configured the simulation.
    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || collision_count_state == nullptr) {
        return;
    }

    // -------------------------------------------------------------------------
    // Step 2. Bind readable local references to the underlying state buffers.
    // -------------------------------------------------------------------------
    //
    // These are references only. No particle data is copied here.
    // The solver continues to operate directly on the original storage owned by
    // the fluid and universe state containers.
    auto& velocities          = velocity_state->data();
    auto& particle_species    = species_state->data();
    auto& number_particle     = number_particle_state->data();
    auto& collision_count     = collision_count_state->data();
    auto& particle_properties = this->_fluid->particle_properties();

    // -------------------------------------------------------------------------
    // Step 3. Convert high-level buffers into raw pointers for device execution.
    // -------------------------------------------------------------------------
    //
    // The device-side lambda passed to atlas::parallel_for needs low-level pointer
    // access to contiguous memory. Wrapper objects or host-only abstractions are
    // generally not suitable for direct capture into device kernels.
    //
    // Important memory note:
    // - mutable_velocity_ptr points to the original velocity buffer and is written
    //   back in place
    // - all other pointers below are read-only views into existing simulation data
    // - no additional per-particle or per-cell memory is allocated here
    auto* mutable_velocity_ptr      = atlas::raw_pointer_cast(velocities.data());
    const auto* species_ptr         = atlas::raw_pointer_cast(particle_species.data());
    const auto* number_particle_ptr = atlas::raw_pointer_cast(number_particle.data());
    const auto* collision_count_ptr = atlas::raw_pointer_cast(collision_count.data());
    const auto* properties_ptr      = atlas::raw_pointer_cast(particle_properties.data());

    // Searcher-owned indexing tables describing how particles are grouped by cell.
    //
    // - indices_ptr     : sorted particle indices after spatial binning
    // - cell_start_ptr  : first sorted-index position belonging to each cell
    // - cell_end_ptr    : one-past-last sorted-index position belonging to each cell
    //
    // These tables allow the solver to interpret each cell as a local particle list
    // without moving the actual particle state buffers themselves.
    const auto* indices_ptr         = this->_searcher->indices();
    const auto* cell_start_ptr      = this->_searcher->cell_start();
    const auto* cell_end_ptr        = this->_searcher->cell_end();

    // Cache small scalar metadata needed repeatedly inside the device kernel.
    //
    // Capturing these by value is cheap and avoids repeated indirect access
    // through solver members from inside the device lambda.
    const int particle_count        = static_cast<int>(this->_fluid->particle_count());
    const int num_of_properties     = static_cast<int>(particle_properties.size());
    const int num_of_cells          = this->_universe->number_of_cells();

    // Copy the collision kernel object locally so it can be captured into the
    // device lambda. This keeps the kernel call self-contained inside the device
    // execution context.
    const auto kernel               = this->_kernel;

    // Optional per-cell solver assignment table.
    //
    // When this pointer is non-null, only cells whose assigned solver id equals
    // the supplied 'index' value will be processed in this pass.
    //
    // This is useful when multiple solvers or multiple passes partition the cell
    // set externally.
    const auto* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    // -------------------------------------------------------------------------
    // Step 4. Launch one device task per universe cell.
    // -------------------------------------------------------------------------
    //
    // Each cell is processed independently, which makes the algorithm naturally
    // parallel at the cell level.
    //
    // Memory behavior inside the device lambda:
    // - reads from species / number-particle / collision-count / property buffers
    // - reads and writes velocity data in place
    // - does not allocate dynamic memory
    // - uses only a few local scalar temporaries and two Vector3<T> values per pair
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // If a solver-assignment table is present, skip any cell that is not
            // assigned to the solver index associated with this invocation.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            // Read the number of particles currently assigned to this cell.
            //
            // Since this solver uses only disjoint consecutive local pairs, the
            // maximum number of usable pairs is floor(count / 2).
            const int count               = static_cast<int>(number_particle_ptr[cell]);
            const int disjoint_pair_count = count / 2;

            // Read the number of collisions requested for this cell by the universe
            // collision-count state.
            const int collisions          = collision_count_ptr[cell];

            // If the cell has fewer than two particles, or if the requested number
            // of collisions is zero or negative, there is nothing to do.
            if (disjoint_pair_count <= 0 || collisions <= 0) {
                return;
            }

            // Fetch the sorted-index interval for the current cell.
            //
            // The pairing logic below interprets this interval as a cell-local
            // sequence of particles after spatial hashing / binning.
            const int begin     = cell_start_ptr[cell];
            const int end       = cell_end_ptr[cell];

            // Clamp the requested collision count so that we never try to use more
            // pairs than the number of disjoint local pairs available in the cell.
            const int collision_limit =
                collisions < disjoint_pair_count ? collisions : disjoint_pair_count;

            // Process only disjoint consecutive local pairs:
            // - pair 0 uses local particle indices 0 and 1
            // - pair 1 uses local particle indices 2 and 3
            // - pair 2 uses local particle indices 4 and 5
            // and so on.
            //
            // This guarantees that within this solver pass, the same local particle
            // is never reused in more than one collision pair for the same cell.
            for (int local_collision = 0; local_collision < collision_limit; ++local_collision) {
                const int lhs_local = local_collision * 2;
                const int rhs_local = lhs_local + 1;

                // Map cell-local particle positions to global particle indices.
                //
                // nth_valid_particle walks the sorted cell interval and resolves the
                // N-th valid particle index, skipping invalid entries if necessary.
                const int particle_i = DsmcSolver<T>::nth_valid_particle(
                    lhs_local,
                    begin,
                    end,
                    particle_count,
                    indices_ptr);

                const int particle_j = DsmcSolver<T>::nth_valid_particle(
                    rhs_local,
                    begin,
                    end,
                    particle_count,
                    indices_ptr);

                // Skip the pair if either local slot failed to resolve to a valid
                // global particle index.
                if (particle_i < 0 || particle_j < 0) {
                    continue;
                }

                // Resolve material/species ids for the two particles.
                //
                // Species ids are later used to index the particle property table
                // required by the collision kernel.
                const std::size_t species_i = species_ptr[particle_i];
                const std::size_t species_j = species_ptr[particle_j];

                // Guard against invalid or out-of-range species ids.
                //
                // Without this check, the collision kernel could read invalid
                // material-property memory.
                if (species_i >= static_cast<std::size_t>(num_of_properties)
                    || species_j >= static_cast<std::size_t>(num_of_properties)) {
                    continue;
                }

                // Load both particle velocities into local temporaries.
                //
                // The kernel operates on local Vector3<T> values, which allows the
                // collision update to happen entirely in registers or local device
                // storage before committing the results back to global memory.
                Vector3<T> lhs_velocity = mutable_velocity_ptr[particle_i];
                Vector3<T> rhs_velocity = mutable_velocity_ptr[particle_j];

                // Apply the selected DSMC collision kernel.
                //
                // The kernel mutates the two local velocity vectors in place using
                // the material properties associated with each particle species.
                kernel(lhs_velocity, rhs_velocity, properties_ptr[species_i], properties_ptr[species_j]);

                // Write the updated post-collision velocities back into the original
                // global velocity buffer.
                //
                // Because this solver uses disjoint local pairs within a cell, the
                // same particle velocity is not written twice by this solver pass
                // for the same cell pairing schedule.
                mutable_velocity_ptr[particle_i] = lhs_velocity;
                mutable_velocity_ptr[particle_j] = rhs_velocity;
            }
        });
}

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
void
DsmcDisjointPairSolver<T>::Builder::validate() const {
    // Validate that every required dependency has been provided before the solver
    // is constructed.
    //
    // This builder cannot produce a usable solver without:
    // - a universe
    // - a fluid
    // - a spatial hashing searcher
    //
    // The kernel type always has a default value, so it does not need a separate
    // null or presence check.
    if (!_universe) {
        throw std::runtime_error("DsmcDisjointPairSolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("DsmcDisjointPairSolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("DsmcDisjointPairSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
DsmcDisjointPairSolver<T>
DsmcDisjointPairSolver<T>::Builder::build() const {
    // Perform the same dependency validation that would otherwise be required
    // at construction sites throughout the codebase.
    validate();

    // Return the solver by value.
    //
    // The returned solver shares ownership of the configured dependencies through
    // the stored host shared pointers. Large simulation buffers are not deep-copied here.
    return DsmcDisjointPairSolver<T>(_universe, _fluid, _searcher, _kernel_type);
}

template <typename T>
atlas::host_shared_ptr<DsmcDisjointPairSolver<T>>
DsmcDisjointPairSolver<T>::Builder::make_host_shared() const {
    // Validate the builder state before allocating the shared solver object.
    validate();

    // Allocate the solver directly in host-shared storage.
    //
    // This is the preferred construction path when the solver will be registered
    // into other runtime systems that expect shared ownership semantics.
    return atlas::make_host_shared<DsmcDisjointPairSolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _kernel_type);
}

} // namespace atlas::system