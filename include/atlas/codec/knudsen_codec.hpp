#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/logging/logging.h>
#include <atlas/math/constants.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

namespace atlas::system {

template <typename T>
typename KnudsenCodec<T>::Builder
KnudsenCodec<T>::builder() noexcept {

    // Return a default-initialized builder for fluent KnudsenCodec construction.
    return Builder {};
}

template <typename T>
KnudsenCodec<T>::KnudsenCodec(UniverseHostPtr<T> domain,
                              FluidHostPtr<T> fluid,
                              SpatialHashingSearcherHostPtr<T> searcher,
                              T characteristic_length)
    : Codec<T>(std::move(domain), std::move(fluid), std::move(searcher))
    , _characteristic_length(characteristic_length) {

    // The characteristic length is a required physical/model parameter and
    // must be strictly positive for Knudsen-number normalization.
    atlas::check<std::invalid_argument>(characteristic_length > T(0))
        << "KnudsenCodec: characteristic_length must be positive.";

    const auto num_of_cells = this->_universe->number_of_cells();

    // Ensure that the universe provides a destination field for encoded
    // Knudsen numbers. Create it lazily if it does not already exist.
    if (!this->_universe->template has_state<atlas::universe::UniverseKnudsenNumberState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseKnudsenNumberState<T>>(
            static_cast<std::size_t>(num_of_cells));
    }

    // Initialize the Knudsen split thresholds used during decode().
    //
    // These define the boundaries between solver allocation regions:
    //   [0, 0.01), [0.01, 0.1), [0.1, 1.0), [1.0, +inf)
    const HostBuffer<T> kn_split { T(0.01), T(0.1), T(1.0) };
    d_kn_split = DeviceBuffer<T>(kn_split.begin(), kn_split.end());
}

template <typename T>
void
KnudsenCodec<T>::encode() {

    // Encode requires:
    // - cell temperature
    // - cell particle count
    // - cell Knudsen-number output state
    auto* temperature_state
        = this->_universe->template state<atlas::universe::UniverseTemperatureState<T>>();
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* knudsen_number_state
        = this->_universe->template state<atlas::universe::UniverseKnudsenNumberState<T>>();

    if (temperature_state == nullptr
        || number_particle_state == nullptr
        || knudsen_number_state == nullptr) {
        return;
    }

    auto& temperature     = temperature_state->data();
    auto& number_particle = number_particle_state->data();
    auto& knudsen_number  = knudsen_number_state->data();

    const auto* temperature_ptr     = atlas::raw_pointer_cast(temperature.data());
    const auto* number_particle_ptr = atlas::raw_pointer_cast(number_particle.data());
    auto* knudsen_number_ptr        = atlas::raw_pointer_cast(knudsen_number.data());
    const int num_of_cells          = this->_universe->number_of_cells();
    const T cell_volume             = this->_universe->cell_volume();
    const T characteristic_length   = _characteristic_length;

    // For each cell, estimate the mean free path from temperature and number density,
    // then normalize it by the configured characteristic length to obtain the
    // Knudsen number.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const T number_density = cell_volume > T(0)
                ? number_particle_ptr[cell] / cell_volume
                : T(0);

            // If the required physical quantities are invalid or degenerate,
            // store zero as a safe fallback.
            if (!(number_density > T(0)) || !(temperature_ptr[cell] > T(0)) || !(characteristic_length > T(0))) {
                knudsen_number_ptr[cell] = T(0);
                return;
            }

            // Mean free path estimate used by this model.
            const T mean_free_path
                = static_cast<T>(atlas::boltzmann_constant) * temperature_ptr[cell] / number_density;

            knudsen_number_ptr[cell] = mean_free_path / characteristic_length;
        });
}

template <typename T>
void
KnudsenCodec<T>::decode() {

    // Decode requires:
    // - encoded Knudsen-number field
    // - split thresholds defining solver regions
    auto* knudsen_number_state
        = this->_universe->template state<atlas::universe::UniverseKnudsenNumberState<T>>();

    if (knudsen_number_state == nullptr || d_kn_split.empty()) {
        return;
    }

    const auto& knudsen_number     = knudsen_number_state->data();
    const auto* knudsen_number_ptr = atlas::raw_pointer_cast(knudsen_number.data());
    const auto* kn_split_ptr       = atlas::raw_pointer_cast(d_kn_split.data());
    auto* allocated_solver_ptr     = atlas::raw_pointer_cast(this->d_allocated_solver.data());
    const int split_count          = static_cast<int>(d_kn_split.size());
    const int num_of_cells         = this->_universe->number_of_cells();

    // For each cell, count how many thresholds the Knudsen number crosses.
    // The resulting count becomes the allocated solver region index.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const T kn           = knudsen_number_ptr[cell];
            int allocated_solver = 0;

            while (allocated_solver < split_count && !(kn < kn_split_ptr[allocated_solver])) {
                ++allocated_solver;
            }

            allocated_solver_ptr[cell] = allocated_solver;
        });
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_domain(UniverseHostPtr<T> domain) noexcept {

    // Store the universe/domain dependency for later construction.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {

    // Store the fluid dependency for later construction.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {

    // Store the spatial hashing searcher dependency for later construction.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_characteristic_length(T characteristic_length) noexcept {

    // Store the characteristic length parameter for later validation and construction.
    _characteristic_length = characteristic_length;
    return *this;
}

template <typename T>
void
KnudsenCodec<T>::Builder::validate() const {

    // All required dependencies must be present before a valid codec can be built.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "KnudsenCodec::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "KnudsenCodec::Builder: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "KnudsenCodec::Builder: searcher must not be null.";

    // The characteristic length must be strictly positive.
    atlas::check<std::invalid_argument>(_characteristic_length > T(0))
        << "KnudsenCodec::Builder: characteristic_length must be positive.";
}

template <typename T>
KnudsenCodec<T>
KnudsenCodec<T>::Builder::build() const {

    // Validate the builder configuration before constructing a value object.
    validate();

    return KnudsenCodec<T>(_domain, _fluid, _searcher, _characteristic_length);
}

template <typename T>
atlas::host_shared_ptr<KnudsenCodec<T>>
KnudsenCodec<T>::Builder::make_host_shared() const {

    // Validate the builder configuration before constructing a shared instance.
    validate();

    return atlas::make_host_shared<KnudsenCodec<T>>(_domain, _fluid, _searcher, _characteristic_length);
}

} // namespace atlas::system