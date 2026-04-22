#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
DsmcNtcSolver<T>::DsmcNtcSolver(UniverseHostPtr<T> universe,
                                FluidHostPtr<T> fluid,
                                SpatialHashingSearcherHostPtr<T> searcher,
                                const DsmcKernelType kernel_type) noexcept
    // Delegate full initialization to the base DsmcSolver class.
    //
    // The base class is responsible for storing the shared simulation
    // dependencies such as:
    //   - the universe, which provides cell-level spatial information,
    //   - the fluid, which owns particle states and material data,
    //   - the spatial searcher, which maps particles into cells,
    //   - the selected DSMC collision kernel type.
    //
    // This derived solver does not add constructor-specific initialization
    // beyond what the base solver already manages.
    : DsmcSolver<T>(std::move(universe), std::move(fluid), std::move(searcher), kernel_type) { }

template <typename T>
typename DsmcNtcSolver<T>::Builder
DsmcNtcSolver<T>::builder() noexcept {
    // Return a new builder instance with default-initialized fields.
    //
    // This provides a fluent construction entry point so the caller can
    // configure the solver step by step before validating and building it.
    return Builder {};
}

template <typename T>
void
DsmcNtcSolver<T>::apply_collisions(const DeviceBuffer<int>* allocated_solver, const int index, const T) {
    // Retrieve the fluid velocity state.
    //
    // This state stores particle velocities and will be both read and written
    // during collision processing because each accepted collision modifies the
    // post-collision velocities of the selected particle pair.
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    // Retrieve the fluid species state.
    //
    // This state maps each particle to its species or material-property index.
    // The species index is later used to access the corresponding particle
    // material properties required by the collision kernel.
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    // Retrieve the universe state that stores the particle count per cell.
    //
    // For each spatial cell, this state tells us how many particles are
    // currently associated with that cell. This count is needed to determine:
    //   - whether collisions are possible at all,
    //   - how many unique particle pairs exist in the cell.
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();

    // Retrieve the universe state that stores the collision attempt count per cell.
    //
    // In the NTC workflow, collision attempts are precomputed elsewhere and
    // stored per cell. This solver consumes that state and performs the actual
    // pairwise collision updates accordingly.
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    // Abort immediately if any required state is missing.
    //
    // Collision processing depends on all of the above states being valid.
    // If even one is unavailable, continuing would either produce incorrect
    // results or require unsafe memory access.
    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || collision_count_state == nullptr) {
        return;
    }

    // Cache references to the underlying containers for readability and to
    // avoid repeatedly traversing accessor chains inside the function.
    auto& velocities          = velocity_state->data();
    auto& particle_species    = species_state->data();
    auto& number_particle     = number_particle_state->data();
    auto& collision_count     = collision_count_state->data();

    // Retrieve the table of per-species material properties from the fluid.
    //
    // Each species entry contains the physical parameters required by the
    // selected DSMC kernel to compute collision behavior.
    auto& particle_properties = this->_fluid->particle_properties();

    // Convert managed containers into raw pointers suitable for device-side use.
    //
    // The device kernel launched by atlas::parallel_for captures these raw
    // pointers by value. Raw pointer access is necessary because higher-level
    // container abstractions may not be directly usable inside device code.
    auto* mutable_velocity_ptr = atlas::raw_pointer_cast(velocities.data());

    // Species indices are read-only during collision application.
    const auto* species_ptr = atlas::raw_pointer_cast(particle_species.data());

    // The per-cell particle count is read-only during this phase.
    const auto* number_particle_ptr = atlas::raw_pointer_cast(number_particle.data());

    // The number of collision attempts for each cell is also read-only here.
    const auto* collision_count_ptr = atlas::raw_pointer_cast(collision_count.data());

    // Material properties are constant lookup data for the duration of the kernel.
    const auto* properties_ptr = atlas::raw_pointer_cast(particle_properties.data());

    // Retrieve the particle index indirection table from the spatial searcher.
    //
    // The searcher stores particles in a cell-organized structure. The indices
    // array is used to translate cell-local ordering into actual particle IDs.
    const auto* indices_ptr = this->_searcher->indices();

    // Retrieve the inclusive start offset of each cell range in the searcher.
    const auto* cell_start_ptr = this->_searcher->cell_start();

    // Retrieve the exclusive end offset of each cell range in the searcher.
    const auto* cell_end_ptr = this->_searcher->cell_end();

    // Cache the global number of particles.
    //
    // This is used when resolving valid particle indices inside the cell-local
    // search range.
    const int particle_count = static_cast<int>(this->_fluid->particle_count());

    // Cache the number of available material-property entries.
    //
    // This is later used to validate species indices before indexing into the
    // property table.
    const int num_of_properties = static_cast<int>(particle_properties.size());

    // Cache the total number of spatial cells in the universe.
    //
    // The device loop will assign one work item per cell.
    const int num_of_cells = this->_universe->number_of_cells();

    // Copy the currently configured collision kernel object.
    //
    // The kernel object encapsulates the actual collision rule and is captured
    // into the device lambda so each cell can apply the selected model.
    const auto kernel = this->_kernel;

    // If a solver-allocation buffer is provided, obtain its raw pointer.
    //
    // This optional buffer allows the caller to partition cells across multiple
    // solver instances. Only cells whose assigned solver index matches the
    // current solver index will be processed.
    const auto* allocated_solver_ptr
        = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;

    // Launch a device-parallel loop over all cells.
    //
    // Each iteration processes collision attempts for exactly one cell.
    // The cell is the natural granularity for the NTC collision stage because
    // collision candidates are defined locally within each spatial bin.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Respect external cell-to-solver partitioning when provided.
            //
            // If the optional allocation array exists and the current cell is
            // assigned to a different solver index, skip all work for this cell.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            // Read the number of particles currently assigned to this cell.
            const int count = static_cast<int>(number_particle_ptr[cell]);

            // Compute the number of unique unordered particle pairs in the cell.
            //
            // For count particles, the number of distinct pairs is:
            //     count * (count - 1) / 2
            //
            // This value is used to deterministically map each collision attempt
            // to one candidate pair ordinal.
            const int pair_count = count * (count - 1) / 2;

            // Read the number of collision attempts scheduled for this cell.
            const int collisions = collision_count_ptr[cell];

            // Skip the cell when no valid collisions are possible.
            //
            // Cases handled here:
            //   - fewer than two particles exist in the cell,
            //   - the number of possible pairs is zero or invalid,
            //   - no collision attempts were scheduled for the cell.
            if (count < 2 || pair_count <= 0 || collisions <= 0) {
                return;
            }

            // Read the searcher range corresponding to the current cell.
            //
            // The range [begin, end) identifies the segment in the searcher's
            // index structure that contains particles belonging to this cell.
            const int begin = cell_start_ptr[cell];
            const int end   = cell_end_ptr[cell];

            // Iterate over all scheduled collision attempts for this cell.
            //
            // The NTC method may request more than one collision attempt in the
            // same cell during a single solver application.
            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                // Initialize the left-hand-side local particle ordinal.
                //
                // The helper function pair_ordinal_to_rhs will update this value
                // so that together with rhs_local it identifies one unique pair
                // corresponding to the selected ordinal.
                int lhs_local = 0;

                // Map the collision attempt index to a valid pair ordinal.
                //
                // The modulo operation wraps the attempt index into the valid
                // range [0, pair_count), ensuring that each attempt references
                // some candidate pair even when collisions exceeds pair_count.
                const int ordinal = local_collision % pair_count;

                // Convert the pair ordinal into two local particle ordinals.
                //
                // The helper computes rhs_local directly and writes lhs_local
                // by reference. Together they identify a unique unordered pair
                // among the particles in the current cell.
                const int rhs_local = DsmcSolver<T>::pair_ordinal_to_rhs(count, ordinal, lhs_local);

                // Convert the local particle ordinal to an actual global particle index.
                //
                // This helper walks the cell-specific search range and resolves
                // the nth valid particle stored there.
                const int particle_i
                    = DsmcSolver<T>::nth_valid_particle(lhs_local, begin, end, particle_count, indices_ptr);

                // Resolve the second particle in the same way.
                const int particle_j
                    = DsmcSolver<T>::nth_valid_particle(rhs_local, begin, end, particle_count, indices_ptr);

                // Skip this collision attempt if either particle resolution failed.
                //
                // Invalid indices may occur if the search range contains holes,
                // invalid entries, or otherwise does not expose enough valid
                // particles for the requested local ordinal.
                if (particle_i < 0 || particle_j < 0) {
                    continue;
                }

                // Read the species indices associated with both particles.
                //
                // These species IDs select the correct material-property records
                // required by the collision kernel.
                const std::size_t species_i = species_ptr[particle_i];
                const std::size_t species_j = species_ptr[particle_j];

                // Validate both species indices before indexing the property table.
                //
                // Out-of-range species values would result in undefined behavior,
                // so such collision attempts are ignored defensively.
                if (species_i >= static_cast<std::size_t>(num_of_properties)
                    || species_j >= static_cast<std::size_t>(num_of_properties)) {
                    continue;
                }

                // Load the current particle velocities into local temporaries.
                //
                // The collision kernel operates on these local copies and updates
                // them in place to represent the post-collision velocities.
                Vector3<T> lhs_velocity = mutable_velocity_ptr[particle_i];
                Vector3<T> rhs_velocity = mutable_velocity_ptr[particle_j];

                // Apply the selected DSMC collision kernel.
                //
                // The kernel uses:
                //   - the two particle velocities,
                //   - the material properties of both species,
                // and modifies the velocity vectors according to the physical
                // collision rule represented by the chosen kernel type.
                kernel(lhs_velocity, rhs_velocity, properties_ptr[species_i], properties_ptr[species_j]);

                // Store the updated post-collision velocity of the first particle.
                mutable_velocity_ptr[particle_i] = lhs_velocity;

                // Store the updated post-collision velocity of the second particle.
                mutable_velocity_ptr[particle_j] = rhs_velocity;
            }
        });
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    // Store the universe dependency inside the builder.
    //
    // The universe provides the spatial discretization and cell-based states
    // required for NTC collision processing.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store the fluid dependency inside the builder.
    //
    // The fluid owns particle states such as velocity and species, as well as
    // the material-property table required by the collision kernel.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    // Store the spatial hashing searcher inside the builder.
    //
    // The searcher is used to map particles into cells and to resolve cell-local
    // particle ordering during pair selection.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    // Store the requested DSMC kernel type inside the builder.
    //
    // This value determines which collision model will be constructed and used
    // by the final solver instance.
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
void
DsmcNtcSolver<T>::Builder::validate() const {
    // Ensure that the universe dependency has been configured.
    //
    // The solver cannot operate without access to the universe and its
    // associated cell-based states.
    if (!_universe) {
        throw std::runtime_error("DsmcNtcSolver::Builder: universe must not be null.");
    }

    // Ensure that the fluid dependency has been configured.
    //
    // The solver requires particle states and material properties from the fluid.
    if (!_fluid) {
        throw std::runtime_error("DsmcNtcSolver::Builder: fluid must not be null.");
    }

    // Ensure that the spatial searcher dependency has been configured.
    //
    // The solver relies on the searcher to obtain cell-local particle ranges
    // and particle index mappings.
    if (!_searcher) {
        throw std::runtime_error("DsmcNtcSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
DsmcNtcSolver<T>
DsmcNtcSolver<T>::Builder::build() const {
    // Validate the builder configuration before constructing the solver.
    //
    // This prevents partially configured solver instances from being created.
    validate();

    // Construct and return the solver by passing the configured dependencies
    // and selected kernel type to the concrete solver constructor.
    return DsmcNtcSolver<T>(_universe, _fluid, _searcher, _kernel_type);
}

template <typename T>
atlas::host_shared_ptr<DsmcNtcSolver<T>>
DsmcNtcSolver<T>::Builder::make_host_shared() const {
    // Validate the builder configuration before allocating a shared solver.
    validate();

    // Construct the solver inside a host-shared smart pointer so ownership can
    // be safely shared across systems that require shared lifetime management.
    return atlas::make_host_shared<DsmcNtcSolver<T>>(_universe, _fluid, _searcher, _kernel_type);
}

} // namespace atlas::system