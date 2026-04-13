#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
typename Fluid<T>::Builder
Fluid<T>::builder() noexcept {
    // Return a fresh builder object for staged `Fluid<T>` construction.
    //
    // The builder path is useful when species definitions, mole fractions,
    // generators, and particle buffer capacity are configured incrementally
    // before the final fluid object is materialized.
    return Builder {};
}

template <typename T>
Fluid<T>::Fluid(const size_t buffer_size)
    : _buffer_size(buffer_size) {
    // Store the maximum particle capacity managed by this fluid instance.
    //
    // This value defines the size of every structure-of-arrays particle buffer.

    // Allocate the particle position buffer.
    //
    // One entry is reserved per possible particle slot.
    d_pos.resize(buffer_size);

    // Allocate the particle velocity buffer.
    d_vel.resize(buffer_size);

    // Allocate the particle temperature buffer.
    d_temperature.resize(buffer_size);

    // Allocate the particle species-id buffer.
    d_species.resize(buffer_size);

    // Allocate the particle active-state buffer.
    //
    // This typically allows runtime code to mark whether a slot currently
    // contains a valid active particle.
    d_active.resize(buffer_size);
}

template <typename T>
int
Fluid<T>::size() const noexcept {
    // Return the number of configured particle species definitions.
    //
    // This is not the active particle count in the simulation buffer.
    // It is the number of species/material entries stored in the fluid metadata.
    return static_cast<int>(_particle_properties.size());
}

template <typename T>
bool
Fluid<T>::empty() const noexcept {
    // A fluid is considered empty when it contains no configured species metadata.
    return size() == 0;
}

template <typename T>
const DeviceBuffer<MatrialProperties<T>>&
Fluid<T>::particles() const noexcept {
    // Return read-only access to the device-resident species/material properties.
    return _particle_properties;
}

template <typename T>
DeviceBuffer<MatrialProperties<T>>&
Fluid<T>::particles() noexcept {
    // Return mutable access to the device-resident species/material properties.
    return _particle_properties;
}

template <typename T>
const DeviceBuffer<T>&
Fluid<T>::mole_fractions() const noexcept {
    // Return read-only access to the device-resident mole-fraction array.
    //
    // The index of each entry corresponds to the same species index used in
    // `_particle_properties` and `_generators`.
    return _mole_fractions;
}

template <typename T>
DeviceBuffer<T>&
Fluid<T>::mole_fractions() noexcept {
    // Return mutable access to the device-resident mole-fraction array.
    return _mole_fractions;
}

template <typename T>
const DeviceBuffer<GenerateOperator<T>>&
Fluid<T>::generators() const noexcept {
    // Return read-only access to the device-resident species generator operators.
    return _generators;
}

template <typename T>
DeviceBuffer<GenerateOperator<T>>&
Fluid<T>::generators() noexcept {
    // Return mutable access to the device-resident species generator operators.
    return _generators;
}

template <typename T>
FluidDeviceProbe<T>
Fluid<T>::make_device_probe() noexcept {
    // Track how many times a fluid device probe has been requested.
    //
    // The runtime expects one canonical probe instance created by the system.
    ++_probe_count;

    // Enforce the invariant that only one authoritative fluid probe exists.
    //
    // This avoids duplicated device-side views that could imply inconsistent
    // ownership or multiple runtime authorities over the same particle buffers.
    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The fluid device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    // Start from a zero-initialized probe object so every exported field has
    // a defined initial state before assignment.
    FluidDeviceProbe<T> probe {};

    // Expose raw device pointers to the species/material metadata buffer.
    probe.particle_property = atlas::raw_pointer_cast(_particle_properties.data());

    // Expose raw device pointers to the particle structure-of-arrays storage.
    probe.pos         = atlas::raw_pointer_cast(d_pos.data());
    probe.vel         = atlas::raw_pointer_cast(d_vel.data());
    probe.temperature = atlas::raw_pointer_cast(d_temperature.data());
    probe.species     = atlas::raw_pointer_cast(d_species.data());
    probe.active      = atlas::raw_pointer_cast(d_active.data());

    // Initialize the active particle count to zero.
    //
    // The probe exposes capacity immediately, but runtime stages are expected
    // to populate and manage the actual live particle count separately.
    probe.particle_count = 0;

    // Publish the maximum particle capacity represented by the SoA buffers.
    probe.buffer_size = _buffer_size;

    return probe;
}

template <typename T>
DeviceBuffer<Vector3<T>>&
Fluid<T>::positions() noexcept {
    // Return mutable access to the particle position buffer.
    return d_pos;
}

template <typename T>
DeviceBuffer<Vector3<T>>&
Fluid<T>::velocities() noexcept {
    // Return mutable access to the particle velocity buffer.
    return d_vel;
}

template <typename T>
DeviceBuffer<T>&
Fluid<T>::temperatures() noexcept {
    // Return mutable access to the particle temperature buffer.
    return d_temperature;
}

template <typename T>
DeviceBuffer<size_t>&
Fluid<T>::species_ids() noexcept {
    // Return mutable access to the particle species-id buffer.
    //
    // Each particle slot can use this to identify which configured species
    // entry in `_particle_properties` it belongs to.
    return d_species;
}

template <typename T>
DeviceBuffer<int>&
Fluid<T>::active() noexcept {
    // Return mutable access to the particle active-flag buffer.
    return d_active;
}

template <typename T>
size_t
Fluid<T>::buffer_size() const noexcept {
    // Return the maximum number of particle slots allocated by this fluid.
    return _buffer_size;
}

template <typename T>
Fluid<T>
Fluid<T>::Builder::build() const {
    // Validate all staged builder data before constructing the fluid object.
    validate();

    // Start from a default-constructed fluid object.
    Fluid<T> f {};

    // Materialize the staged species/material metadata into device-resident storage.
    f._particle_properties = DeviceBuffer<MatrialProperties<T>>(_particles.begin(), _particles.end());

    // Materialize the staged mole fractions into device-resident storage.
    f._mole_fractions = DeviceBuffer<T>(_mole_fractions.begin(), _mole_fractions.end());

    // Materialize the staged generator operators into device-resident storage.
    f._generators = DeviceBuffer<GenerateOperator<T>>(_generators.begin(), _generators.end());

    // Store the configured particle-capacity ceiling.
    f._buffer_size = _buffer_size;

    // Allocate particle structure-of-arrays storage to match the configured capacity.
    //
    // These buffers hold runtime particle state independently of the number of
    // configured species definitions.
    f.d_pos.resize(_buffer_size);
    f.d_vel.resize(_buffer_size);
    f.d_temperature.resize(_buffer_size);
    f.d_species.resize(_buffer_size);
    f.d_active.resize(_buffer_size);

    return f;
}

template <typename T>
atlas::host_shared_ptr<Fluid<T>>
Fluid<T>::Builder::make_host_shared() const {
    // Build the fluid by value first.
    auto f = build();

    // Move the built fluid into host-shared managed storage and return the handle.
    return atlas::make_host_shared<Fluid<T>>(std::move(f));
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(const MatrialProperties<T>& p) {
    // Add one species using the default mole fraction of 1.
    //
    // This is a convenience overload that forwards to the more general path.
    return add_species(p, T(1));
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(const MatrialProperties<T>& p, T mole_fraction) {
    // Add one species with an explicit mole fraction and no custom generator.
    return add_species(p, mole_fraction, nullptr);
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(const MatrialProperties<T>& p,
                               T mole_fraction,
                               GeneratorHostPtr<T> generator) {
    // Append the species material properties to the builder's staged list.
    _particles.push_back(p);

    // Append the mole fraction at the matching species index.
    _mole_fractions.push_back(mole_fraction);

    // Append the generation operator at the matching species index.
    //
    // If no generator is provided, store a default-constructed operator so
    // all builder-side arrays remain index-aligned.
    _generators.push_back(generator ? generator->make_generate_operator() : GenerateOperator<T> {});

    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(MatrialPropertiesHostPtr<T> p) {
    // Add one species from a host pointer using the default mole fraction of 1.
    return add_species(std::move(p), T(1));
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(MatrialPropertiesHostPtr<T> p, T mole_fraction) {
    // Add one species from a host pointer with an explicit mole fraction
    // and no custom generator.
    return add_species(std::move(p), mole_fraction, nullptr);
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(MatrialPropertiesHostPtr<T> p,
                               T mole_fraction,
                               GeneratorHostPtr<T> generator) {
    // Optionally reject null species pointers when strict pointer validity
    // is requested by the builder configuration.
    if (_reject_null_particles && !p) {
        throw std::runtime_error(
            "Fluid::Builder: null particle pointer encountered.");
    }

    // Store either:
    // - the dereferenced material properties if the pointer is valid, or
    // - a default-constructed material properties object if null pointers are allowed.
    //
    // This keeps species-array indices aligned regardless of pointer validity policy.
    _particles.push_back(p ? *p : MatrialProperties<T> {});

    // Store the mole fraction for the same species entry.
    _mole_fractions.push_back(mole_fraction);

    // Store the corresponding generator operator, or a default one when absent.
    _generators.push_back(generator ? generator->make_generate_operator() : GenerateOperator<T> {});

    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps) {
    // Add a batch of species values, assigning each the default mole fraction of 1.
    const int n = static_cast<int>(ps.size());
    for (int i = 0; i < n; ++i) {
        add_species(ps[i], T(1));
    }
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialProperties<T>>& ps,
    const HostBuffer<T>& mole_fractions) {
    // Add a batch of species values with explicit mole fractions and no generators.
    //
    // Create a same-sized null-generator array so the fully general bulk path
    // can be reused.
    const HostBuffer<GeneratorHostPtr<T>> generators(ps.size(), nullptr);
    return add_species_bulk(ps, mole_fractions, generators);
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialProperties<T>>& ps,
    const HostBuffer<T>& mole_fractions,
    const HostBuffer<GeneratorHostPtr<T>>& generators) {
    const int n = static_cast<int>(ps.size());

    // Require all parallel input arrays to have identical length so each species
    // has exactly one matching mole fraction and one matching generator entry.
    if (static_cast<int>(mole_fractions.size()) != n || static_cast<int>(generators.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_species_bulk(values, mole_fractions, generators): size mismatch.");
    }

    // Reuse the scalar add_species path so storage behavior and any per-item
    // handling remain centralized in one place.
    for (int i = 0; i < n; ++i) {
        add_species(ps[i], mole_fractions[i], generators[i]);
    }
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialPropertiesHostPtr<T>>& ps) {
    // Add a batch of species pointers, assigning each the default mole fraction of 1.
    const int n = static_cast<int>(ps.size());
    for (int i = 0; i < n; ++i) {
        add_species(ps[i], T(1));
    }
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialPropertiesHostPtr<T>>& ps,
    const HostBuffer<T>& mole_fractions) {
    // Add a batch of species pointers with explicit mole fractions and no generators.
    const HostBuffer<GeneratorHostPtr<T>> generators(ps.size(), nullptr);
    return add_species_bulk(ps, mole_fractions, generators);
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialPropertiesHostPtr<T>>& ps,
    const HostBuffer<T>& mole_fractions,
    const HostBuffer<GeneratorHostPtr<T>>& generators) {
    const int n = static_cast<int>(ps.size());

    // Require all parallel input arrays to have identical length so pointer-based
    // species input remains index-aligned with mole fractions and generators.
    if (static_cast<int>(mole_fractions.size()) != n || static_cast<int>(generators.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_species_bulk(ptrs, mole_fractions, generators): size mismatch.");
    }

    // Reuse the scalar pointer-based add_species path so null-pointer handling
    // and generator conversion stay centralized.
    for (int i = 0; i < n; ++i) {
        add_species(ps[i], mole_fractions[i], generators[i]);
    }
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_buffer_size(const size_t buffer_size) noexcept {
    // Store the desired particle-capacity ceiling in the builder.
    _buffer_size = buffer_size;
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::require_non_empty(bool on) noexcept {
    // Control whether validation should reject a fluid with zero configured species.
    _require_non_empty = on;
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::reject_null_particles(bool on) noexcept {
    // Control whether pointer-based species insertion should reject null pointers
    // instead of mapping them to default-constructed material properties.
    _reject_null_particles = on;
    return *this;
}

template <typename T>
void
Fluid<T>::Builder::validate() const {
    // Ensure all builder-side parallel arrays remain perfectly aligned.
    //
    // Each species entry must have exactly:
    // - one material-properties entry,
    // - one mole-fraction entry,
    // - one generator entry.
    if (_particles.size() != _mole_fractions.size() || _particles.size() != _generators.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/mole_fractions/generators size mismatch.");
    }

    // Optionally require at least one configured species.
    if (_require_non_empty && _particles.empty()) {
        throw std::runtime_error(
            "Fluid::Builder: fluid must contain at least one species.");
    }

    // Copy staged mole fractions into a host buffer for validation and normalization.
    const HostBuffer<T> mole_fractions_host(_mole_fractions.begin(), _mole_fractions.end());

    // Accumulate the total mole fraction while checking each entry.
    T sum       = T(0);
    const int n = static_cast<int>(mole_fractions_host.size());

    for (int i = 0; i < n; ++i) {
        // Every mole fraction must be finite and non-negative.
        if (!std::isfinite(mole_fractions_host[i]) || mole_fractions_host[i] < T(0)) {
            throw std::runtime_error(
                "Fluid::Builder: mole fractions must be finite and non-negative.");
        }

        sum += mole_fractions_host[i];
    }

    if (n > 0 && sum > T(0)) {
        // Normalize the mole fractions so the stored composition sums to 1.
        //
        // This ensures runtime code sees a valid species composition regardless
        // of the scale of user-provided positive weights.
        HostBuffer<T> normalized = mole_fractions_host;
        for (int i = 0; i < n; ++i) {
            normalized[i] /= sum;
        }

        // Write the normalized composition back into the builder's staged storage.
        //
        // The cast is used because validation mutates the internal staged state
        // to guarantee that the later build step sees normalized fractions.
        auto& mole_fractions = const_cast<DeviceBuffer<T>&>(_mole_fractions);
        mole_fractions = DeviceBuffer<T>(normalized.begin(), normalized.end());
    } else if (n > 0) {
        // Reject the case where mole fractions exist but their total weight is zero,
        // because normalization would be undefined and the composition meaningless.
        throw std::runtime_error(
            "Fluid::Builder: mole fractions sum to zero.");
    }
}

} // namespace atlas::system