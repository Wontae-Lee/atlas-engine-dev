#include <atlas/solver/sph/sph_solver.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/solver/detail/solver_probe_common.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

namespace {

bool
has_particle_states(const FluidHostPtr& fluid) noexcept {
    return fluid != nullptr
        && fluid->state<FluidPositionState>() != nullptr
        && fluid->state<FluidVelocityState>() != nullptr
        && fluid->state<FluidSpeciesState>() != nullptr;
}

bool
sph_probe_ready(const UniverseHostPtr& universe,
                const FluidHostPtr& fluid,
                const SearcherHostPtr& searcher) noexcept {
    return universe != nullptr
        && searcher != nullptr
        && has_particle_states(fluid)
        && universe->state<UniverseNumberParticleState>() != nullptr
        && universe->state<UniverseFieldForceState>() != nullptr;
}

}

SphSolver::SphSolver(UniverseHostPtr universe,
                     FluidHostPtr fluid,
                     SearcherHostPtr searcher,
                     const SphKernelType kernel_type) noexcept
    : Solver(std::move(universe), std::move(fluid), std::move(searcher)) {

    _kernel = SphKernel(kernel_type);
    ensure_states();
}

SphSolver::Builder
SphSolver::builder() noexcept {

    return Builder {};
}

SphKernelType
SphSolver::kernel_type() const noexcept {

    return _kernel.type;
}

void
SphSolver::solve(const float dt) {

    if (!initialize_context()) {
        return;
    }

    if (!(dt > 0.0f)) {
        throw std::invalid_argument("SphSolver: dt must be positive.");
    }

    if (!prepare_fields()) {
        reset_fields();
        return;
    }

    if (!make_probe()) {
        return;
    }

    update();

    accelerate(dt);
}

void
SphSolver::solve(const DeviceBuffer<int>*, const int, const float) {
}

void
SphSolver::ensure_states() {
    if (!this->_universe) {
        return;
    }

    const auto count = static_cast<std::size_t>(this->_universe->cell_count());

    if (auto* state = this->_universe->state<UniverseNumberParticleState>();
        state == nullptr) {
        this->_universe->emplace_state<UniverseNumberParticleState>(count);
    } else if (state->size() != count) {
        state->data().resize(count);
    }

    if (auto* state = this->_universe->state<UniverseFieldForceState>();
        state == nullptr) {
        this->_universe->emplace_state<UniverseFieldForceState>(count);
    } else if (state->size() != count) {
        state->data().resize(count);
    }
}

bool
SphSolver::initialize_context() noexcept {

    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_fields();
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        return false;
    }

    if (!has_particle_states(this->_fluid)) {
        reset_fields();
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        return false;
    }

    ensure_states();
    this->_searcher->build();

    return true;
}

bool
SphSolver::make_probe() noexcept {
    _probe = {};

    if (!sph_probe_ready(this->_universe, this->_fluid, this->_searcher)) {
        return false;
    }

    detail::fill_common_solver_probe(_probe, this->_universe, this->_fluid, this->_searcher);

    _probe.position_ptr         = atlas::raw_pointer_cast(this->_fluid->state<FluidPositionState>()->data().data());
    _probe.field_force_ptr      = atlas::raw_pointer_cast(this->_universe->state<UniverseFieldForceState>()->data().data());
    _probe.neighbor_offsets_ptr = this->_searcher->neighbor_offsets();
    _probe.neighbor_indices_ptr = this->_searcher->neighbor_indices();
    _probe.lower_corner         = this->_searcher->lower_corner();
    _probe.grid_size            = this->_searcher->grid_size();
    _probe.inverse_cell_size    = this->_searcher->inverse_cell_size();
    _probe.cell_size            = this->_searcher->cell_size();
    _probe.particle_count       = static_cast<int>(this->_fluid->particle_count());
    _probe.cell_count           = this->_universe->cell_count();
    _probe.property_count       = static_cast<int>(this->_fluid->particle_properties().size());
    _probe.kernel               = _kernel;

    return true;
}

bool
SphSolver::prepare_fields() {

    const int particle_count = static_cast<int>(this->_fluid->particle_count());
    if (particle_count <= 0) {
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        reset_fields();
        return false;
    }

    _density.resize(static_cast<std::size_t>(particle_count));
    _pressure.resize(static_cast<std::size_t>(particle_count));
    _acceleration.resize(static_cast<std::size_t>(particle_count));

    reset_fields();

    return true;
}

void
SphSolver::reset_fields() {
    if (!this->_universe) {
        return;
    }

    ensure_states();
    this->_universe->state<UniverseNumberParticleState>()->reset();
    this->_universe->state<UniverseFieldForceState>()->reset();
}

void
SphSolver::update() {

    estimate_density();
    count_particles();
}

void
SphSolver::estimate_density() {
    const auto probe   = _probe;
    auto* density_ptr  = atlas::raw_pointer_cast(_density.data());
    auto* pressure_ptr = atlas::raw_pointer_cast(_pressure.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [=] ATLAS_ALL_DEVICE(const int particle_index) {
            const std::size_t species_index = probe.species_ptr[particle_index];
            if (species_index >= static_cast<std::size_t>(probe.property_count)) {
                density_ptr[particle_index]  = 0.0f;
                pressure_ptr[particle_index] = 0.0f;
                return;
            }

            const auto& property  = probe.properties_ptr[species_index];
            const float h         = probe.cell_size;
            const float h_squared = h * h;
            const float rho0      = SphSolver::rest_density(property);
            const float k         = SphSolver::pressure_coefficient(property);
            const Float3 position = probe.position_ptr[particle_index];
            // density_weight(0, h) is the particle's own contribution to its
            // density estimate (r=0, i.e. a particle always "sees" itself)
            // — the standard SPH self-term, added before summing neighbors.
            float density         = property.mass * probe.kernel.density_weight(0.0f, h);

            const int begin = probe.neighbor_offsets_ptr[particle_index];
            const int end   = probe.neighbor_offsets_ptr[particle_index + 1];

            for (int neighbor_offset = begin; neighbor_offset < end; ++neighbor_offset) {
                const int neighbor_index = probe.neighbor_indices_ptr[neighbor_offset];

                if (neighbor_index < 0 || neighbor_index >= probe.particle_count) {
                    continue;
                }

                const Float3 delta         = position - probe.position_ptr[neighbor_index];
                const float radius_squared = delta.length_squared();

                if (radius_squared > h_squared) {
                    continue;
                }

                const float radius = atlas::sqrt_nonnegative(radius_squared);

                const std::size_t neighbor_species_index = probe.species_ptr[neighbor_index];

                if (neighbor_species_index >= static_cast<std::size_t>(probe.property_count)) {
                    continue;
                }

                const float neighbor_mass = probe.properties_ptr[neighbor_species_index].mass;

                density += neighbor_mass * probe.kernel.density_weight(radius, h);
            }

            // An isolated particle with no self-term contribution (density
            // came out <= 0, a degenerate case) falls back to the rest
            // density instead of producing a zero/negative pressure below.
            if (!(density > 0.0f)) {
                density = rho0;
            }

            density_ptr[particle_index]  = density;
            pressure_ptr[particle_index] = k * (density - rho0);
        });
}

void
SphSolver::count_particles() {
    const auto probe = _probe;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (begin < 0 || end <= begin) {
                probe.number_particle_ptr[cell] = 0.0f;
                return;
            }

            probe.number_particle_ptr[cell] = static_cast<float>(end - begin);
        });
}

void
SphSolver::accelerate(const float dt) {
    const auto probe         = _probe;
    const auto* density_ptr  = atlas::raw_pointer_cast(_density.data());
    const auto* pressure_ptr = atlas::raw_pointer_cast(_pressure.data());
    auto* acceleration_ptr   = atlas::raw_pointer_cast(_acceleration.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [=] ATLAS_ALL_DEVICE(const int particle_index) {
            const std::size_t species_index = probe.species_ptr[particle_index];
            if (species_index >= static_cast<std::size_t>(probe.property_count)) {
                acceleration_ptr[particle_index] = Float3(0.0f, 0.0f, 0.0f);
                return;
            }

            const auto& property  = probe.properties_ptr[species_index];
            const float h         = probe.cell_size;
            const float h_squared = h * h;
            const float mu        = property.dynamic_viscosity.value_or(0.0f);
            const float mass      = property.mass;
            const Float3 position = probe.position_ptr[particle_index];
            const Float3 velocity = probe.velocity_ptr[particle_index];
            Float3 acceleration(0.0f, 0.0f, 0.0f);

            const int begin = probe.neighbor_offsets_ptr[particle_index];
            const int end   = probe.neighbor_offsets_ptr[particle_index + 1];

            for (int neighbor_offset = begin; neighbor_offset < end; ++neighbor_offset) {
                const int neighbor_index = probe.neighbor_indices_ptr[neighbor_offset];

                if (neighbor_index < 0 || neighbor_index >= probe.particle_count) {
                    continue;
                }

                const std::size_t neighbor_species_index = probe.species_ptr[neighbor_index];

                if (neighbor_species_index >= static_cast<std::size_t>(probe.property_count)) {
                    continue;
                }

                const auto& neighbor_property = probe.properties_ptr[neighbor_species_index];

                const float neighbor_mass    = neighbor_property.mass;
                const float neighbor_density = density_ptr[neighbor_index];

                if (!(neighbor_density > 0.0f)) {
                    continue;
                }

                const Float3 delta         = position - probe.position_ptr[neighbor_index];
                const float radius_squared = delta.length_squared();

                if (!(radius_squared > 0.0f) || radius_squared > h_squared) {
                    continue;
                }

                const float radius = atlas::sqrt_nonnegative(radius_squared);

                const Float3 grad = probe.kernel.pressure_gradient(delta, radius, h);

                // Symmetrized pressure force (Muller et al. 2003 form):
                // averaging this particle's and the neighbor's pressure
                // before dividing by the neighbor's density keeps the force
                // antisymmetric between a pair (since grad(W) itself flips
                // sign when delta flips), without needing the fully
                // symmetric P_i/rho_i^2 + P_j/rho_j^2 form — see
                // sph_solver.h's top-of-file documentation.
                const float pressure_term = (pressure_ptr[particle_index] + pressure_ptr[neighbor_index])
                    / (2.0f * neighbor_density);

                acceleration -= grad * (neighbor_mass * pressure_term);

                // Viscous force only computed when the material actually
                // has viscosity — skips a kernel evaluation and a division
                // for the (common) inviscid case.
                if (mu > 0.0f) {
                    const float laplacian = probe.kernel.viscosity_laplacian(radius, h);

                    acceleration += (probe.velocity_ptr[neighbor_index] - velocity)
                        * (mu * neighbor_mass * laplacian / neighbor_density);
                }
            }

            if (!(mass > 0.0f)) {
                acceleration = Float3(0.0f, 0.0f, 0.0f);
            }

            // Semi-implicit Euler: velocity updated here from this step's
            // acceleration; position integration is left to a separate
            // system step (see sph_solver.h's top-of-file documentation).
            acceleration_ptr[particle_index]   = acceleration;
            probe.velocity_ptr[particle_index] = velocity + acceleration * dt;
        });

    // Second pass: average this step's per-particle accelerations into a
    // coarse per-cell diagnostic field, for callers that want an
    // approximate force sample without re-deriving it per particle.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (begin < 0 || end <= begin) {
                probe.field_force_ptr[cell] = Float3(0.0f, 0.0f, 0.0f);
                return;
            }

            Float3 accumulated_force(0.0f, 0.0f, 0.0f);
            int count = 0;

            for (int sorted_index = begin; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                const std::size_t species_index = probe.species_ptr[particle_index];

                if (species_index >= static_cast<std::size_t>(probe.property_count)) {
                    continue;
                }

                accumulated_force += acceleration_ptr[particle_index] * probe.properties_ptr[species_index].mass;

                ++count;
            }

            probe.field_force_ptr[cell] = count > 0 ? accumulated_force / static_cast<float>(count)
                                                    : Float3(0.0f, 0.0f, 0.0f);
        });
}

SphSolver::Builder&
SphSolver::Builder::with_universe(UniverseHostPtr universe) noexcept {

    _universe = std::move(universe);
    return *this;
}

SphSolver::Builder&
SphSolver::Builder::with_fluid(FluidHostPtr fluid) noexcept {

    _fluid = std::move(fluid);
    return *this;
}

SphSolver::Builder&
SphSolver::Builder::with_searcher(SearcherHostPtr searcher) noexcept {

    _searcher = std::move(searcher);
    return *this;
}

SphSolver::Builder&
SphSolver::Builder::with_kernel_type(const SphKernelType kernel_type) noexcept {

    _kernel_type = kernel_type;
    return *this;
}

void
SphSolver::Builder::validate() const {

    if (!_universe) {
        throw std::runtime_error("SphSolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("SphSolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("SphSolver::Builder: searcher must not be null.");
    }
}

SphSolver
SphSolver::Builder::build() const {

    validate();
    return SphSolver(_universe, _fluid, _searcher, _kernel_type);
}

atlas::host_shared_ptr<SphSolver>
SphSolver::Builder::make_host_shared() const {

    validate();
    return atlas::make_host_shared<SphSolver>(_universe, _fluid, _searcher, _kernel_type);
}

}
