#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/random/seed.h>

#include <cstddef>
#include <utility>

namespace atlas {

// Emits Maxwell-Boltzmann distributed velocities using the per-species mass
// (derived from a MaterialDictionary or supplied directly), tagging each
// particle with a species sampled from its ratio table and writing temperature.
class MaxwellBoltzmannGenerator final {
public:
    class Builder;

public:
    MaxwellBoltzmannGenerator() = default;

    ATLAS_HOST
    MaxwellBoltzmannGenerator(DeviceBuffer<float> species_ratios,
                              DeviceBuffer<float> species_numbers,
                              DeviceBuffer<float> species_mass,
                              float temperature,
                              Float3 bulk_velocity,
                              unsigned int seed) noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    set_bulk_velocity(const Float3& bulk_velocity) noexcept {
        _bulk_velocity = bulk_velocity;
    }

    ATLAS_NODISCARD ATLAS_HOST Float3
    bulk_velocity() const noexcept {
        return _bulk_velocity;
    }

    ATLAS_NODISCARD ATLAS_HOST float
    temperature() const noexcept {
        return _temperature;
    }

    ATLAS_NODISCARD ATLAS_HOST int
    generate(FluidVelocityState* velocities,
             FluidSpeciesState* species,
             std::size_t offset,
             std::size_t count) const;

private:
    DeviceBuffer<float> _species_ratios;

    DeviceBuffer<float> _species_numbers;

    DeviceBuffer<float> _species_mass;

    float _temperature { 273.15f };

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f };

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED };
};

class MaxwellBoltzmannGenerator::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_species_ratios(const HostBuffer<float>& species_ratios);

    ATLAS_HOST Builder&
    with_species_numbers(const HostBuffer<float>& species_numbers);

    // Per-species mass is derived by looking each species number up in the
    // dictionary, or supplied directly via with_species_mass. Only the mass of
    // each material is taken; the dictionary itself is not retained.
    ATLAS_HOST Builder&
    with_material_dictionary(const MaterialDictionary& material_dictionary);

    ATLAS_HOST Builder&
    with_species_mass(const HostBuffer<float>& species_mass);

    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

    ATLAS_HOST Builder&
    with_bulk_velocity(const Float3& bulk_velocity) noexcept;

    ATLAS_HOST Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_NODISCARD ATLAS_HOST MaxwellBoltzmannGenerator
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<MaxwellBoltzmannGenerator>
    make_host_shared();

private:
    ATLAS_HOST HostBuffer<float>
    resolve_species_mass() const;

    ATLAS_HOST void
    validate() const;

private:
    HostBuffer<float> _species_ratios;

    HostBuffer<float> _species_numbers;

    HostBuffer<float> _species_mass;

    // Mass of every material in the dictionary, indexed by species id.
    HostBuffer<float> _material_mass;

    float _temperature { 273.15f };

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f };

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED };
};

using MaxwellBoltzmannGeneratorHostPtr = atlas::host_shared_ptr<MaxwellBoltzmannGenerator>;

using MaxwellBoltzmannGeneratorDevicePtr = atlas::device_shared_ptr<MaxwellBoltzmannGenerator>;

}
