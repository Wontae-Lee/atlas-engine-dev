#include <atlas/codec/knudsen_codec.h>

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas {

KnudsenCodec::Builder
KnudsenCodec::builder() noexcept {
    return Builder {};
}

KnudsenCodec::KnudsenCodec(const float representative_characteristic_length,
                           const float representative_collision_cross_sectional_area,
                           const float representative_statistical_weight,
                           const float representative_cell_volume)
    : _representative_characteristic_length(representative_characteristic_length)
    , _representative_collision_cross_sectional_area(representative_collision_cross_sectional_area)
    , _representative_statistical_weight(representative_statistical_weight)
    , _representative_cell_volume(representative_cell_volume) {
}

void
KnudsenCodec::allocate(const UniverseTemperatureState*, // temperature: unused by this leaf
                       const UniverseNumberParticleState* number_particle,
                       UniverseAllocatedSolverState* allocated_solver) const {
    // Nothing to read or write into: treat a null input or output as a no-op.
    if (number_particle == nullptr || allocated_solver == nullptr) {
        return;
    }

    const int cell_count = static_cast<int>(allocated_solver->size());

    // Refuse to run on an empty grid or when the two per-cell buffers disagree in
    // length; the codec never resizes the output, so a mismatch would be unsafe.
    if (cell_count == 0 || number_particle->size() != allocated_solver->size()) {
        return;
    }

    // Drop the thrust device_vector wrappers to raw device pointers so the extended
    // lambda below can index them from GPU threads.
    const auto* number_particle_ptr = atlas::raw_pointer_cast(number_particle->data().data());
    auto* allocated_solver_ptr      = atlas::raw_pointer_cast(allocated_solver->data().data());

    // Copy the whole (trivially-copyable) leaf by value so the device lambda captures
    // its representative scalars and split table without dereferencing `this` on the GPU.
    const KnudsenCodec codec = *this;

    // One thread per cell: estimate the Knudsen number from the cell's particle count
    // and bucket it into a solver index, writing the result in place.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            allocated_solver_ptr[cell] = codec.solver_index(
                codec.knudsen_number(number_particle_ptr[cell]));
        });
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_representative_characteristic_length(
    const float representative_characteristic_length) noexcept {
    _representative_characteristic_length = representative_characteristic_length;
    return *this;
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_representative_collision_cross_sectional_area(
    const float representative_collision_cross_sectional_area) noexcept {
    _representative_collision_cross_sectional_area = representative_collision_cross_sectional_area;
    return *this;
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_representative_statistical_weight(
    const float representative_statistical_weight) noexcept {
    _representative_statistical_weight = representative_statistical_weight;
    return *this;
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_representative_cell_volume(
    const float representative_cell_volume) noexcept {
    _representative_cell_volume = representative_cell_volume;
    return *this;
}

void
KnudsenCodec::Builder::validate() const {
    atlas::check<std::invalid_argument>(_representative_characteristic_length > 0.0f)
        << "KnudsenCodec::Builder: representative_characteristic_length must be positive.";
    atlas::check<std::invalid_argument>(_representative_collision_cross_sectional_area > 0.0f)
        << "KnudsenCodec::Builder: representative_collision_cross_sectional_area must be positive.";
    atlas::check<std::invalid_argument>(_representative_statistical_weight > 0.0f)
        << "KnudsenCodec::Builder: representative_statistical_weight must be positive.";
    atlas::check<std::invalid_argument>(_representative_cell_volume > 0.0f)
        << "KnudsenCodec::Builder: representative_cell_volume must be positive.";
}

KnudsenCodec
KnudsenCodec::Builder::build() const {
    validate();

    return KnudsenCodec(
        _representative_characteristic_length,
        _representative_collision_cross_sectional_area,
        _representative_statistical_weight,
        _representative_cell_volume);
}

atlas::host_shared_ptr<KnudsenCodec>
KnudsenCodec::Builder::make_host_shared() const {
    return atlas::make_host_shared<KnudsenCodec>(build());
}

}
