#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/random/seed.h>

#include <cstddef>
#include <utility>

namespace atlas {

// Emits normally-distributed velocities with a fixed standard deviation (sigma),
// tagging each particle with a species sampled from its per-species ratio table
// and writing its temperature.
class MaxwellSigmaGenerator final {
public:
    class Builder;

public:
    MaxwellSigmaGenerator() = default;

    ATLAS_HOST
    MaxwellSigmaGenerator(DeviceBuffer<float> species_ratios,
                          DeviceBuffer<float> species_numbers,
                          float temperature,
                          float sigma,
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

    float _temperature { 273.15f };

    float _sigma { 0.0f };

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f };

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED };
};

class MaxwellSigmaGenerator::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_species_ratios(const HostBuffer<float>& species_ratios);

    ATLAS_HOST Builder&
    with_species_numbers(const HostBuffer<float>& species_numbers);

    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

    ATLAS_HOST Builder&
    with_sigma(float sigma) noexcept;

    ATLAS_HOST Builder&
    with_bulk_velocity(const Float3& bulk_velocity) noexcept;

    ATLAS_HOST Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_NODISCARD ATLAS_HOST MaxwellSigmaGenerator
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<MaxwellSigmaGenerator>
    make_host_shared();

private:
    ATLAS_HOST void
    validate() const;

private:
    HostBuffer<float> _species_ratios;

    HostBuffer<float> _species_numbers;

    float _temperature { 273.15f };

    float _sigma { 0.0f };

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f };

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED };
};

using MaxwellSigmaGeneratorHostPtr = atlas::host_shared_ptr<MaxwellSigmaGenerator>;

using MaxwellSigmaGeneratorDevicePtr = atlas::device_shared_ptr<MaxwellSigmaGenerator>;

}
