#include <atlas/generator/maxwell_sigma_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace atlas {

MaxwellSigmaGenerator::MaxwellSigmaGenerator(DeviceBuffer<float> species_ratios,
                                             DeviceBuffer<float> species_numbers,
                                             const float temperature,
                                             const float sigma,
                                             const unsigned int seed) noexcept
    : _species_ratios(std::move(species_ratios))
    , _species_numbers(std::move(species_numbers))
    , _temperature(temperature)
    , _sigma(sigma)
    , _seed(seed) {
}

MaxwellSigmaGenerator::Builder
MaxwellSigmaGenerator::builder() noexcept {
    return Builder {};
}

int
MaxwellSigmaGenerator::generate(FluidVelocityState* velocities,
                                FluidTemperatureState* temperatures,
                                FluidSpeciesState* species,
                                const std::size_t offset,
                                const std::size_t count) const {
    if (velocities == nullptr || temperatures == nullptr || species == nullptr || count == 0) {
        return 0;
    }

    DeviceBuffer<Float3>&      velocity_buffer    = velocities->data();
    DeviceBuffer<float>&       temperature_buffer = temperatures->data();
    DeviceBuffer<std::size_t>& species_buffer     = species->data();

    const std::size_t capacity = std::min(velocity_buffer.size(),
                                          std::min(temperature_buffer.size(), species_buffer.size()));
    if (offset >= capacity) {
        return 0;
    }

    const std::size_t writable = std::min(count, capacity - offset);
    if (writable == 0) {
        return 0;
    }

    const float        temperature   = _temperature;
    const float        sigma         = _sigma;
    const unsigned int seed          = _seed;
    const int          species_count = static_cast<int>(_species_ratios.size());

    const float* ratios          = atlas::raw_pointer_cast(_species_ratios.data());
    const float* numbers         = atlas::raw_pointer_cast(_species_numbers.data());
    Float3*      velocity_ptr    = atlas::raw_pointer_cast(velocity_buffer.data());
    float*       temperature_ptr = atlas::raw_pointer_cast(temperature_buffer.data());
    std::size_t* species_ptr     = atlas::raw_pointer_cast(species_buffer.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        writable,
        [=] ATLAS_ALL_DEVICE(const std::size_t i) {
            const std::size_t   index = offset + i;
            const std::uint64_t key   = atlas::shuffle_key(
                static_cast<int>(index),
                static_cast<std::uint64_t>(seed));

            atlas::default_random_engine engine(static_cast<unsigned int>(key));

            species_ptr[index]     = atlas::sample_weighted_choice(ratios, numbers, species_count, engine);
            temperature_ptr[index] = temperature;
            velocity_ptr[index]    = (sigma > 0.0f)
                ? atlas::sample_normal_vector(engine, sigma)
                : Float3(0.0f, 0.0f, 0.0f);
        });

    return static_cast<int>(writable);
}

MaxwellSigmaGenerator::Builder&
MaxwellSigmaGenerator::Builder::with_species_ratios(const HostBuffer<float>& species_ratios) {
    _species_ratios = species_ratios;
    return *this;
}

MaxwellSigmaGenerator::Builder&
MaxwellSigmaGenerator::Builder::with_species_numbers(const HostBuffer<float>& species_numbers) {
    _species_numbers = species_numbers;
    return *this;
}

MaxwellSigmaGenerator::Builder&
MaxwellSigmaGenerator::Builder::with_temperature(const float temperature) noexcept {
    _temperature = temperature;
    return *this;
}

MaxwellSigmaGenerator::Builder&
MaxwellSigmaGenerator::Builder::with_sigma(const float sigma) noexcept {
    _sigma = sigma;
    return *this;
}

MaxwellSigmaGenerator::Builder&
MaxwellSigmaGenerator::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

MaxwellSigmaGenerator
MaxwellSigmaGenerator::Builder::build() {
    validate();

    MaxwellSigmaGenerator generator(
        DeviceBuffer<float>(_species_ratios.begin(), _species_ratios.end()),
        DeviceBuffer<float>(_species_numbers.begin(), _species_numbers.end()),
        _temperature,
        _sigma,
        _seed);

    _species_ratios.clear();
    _species_numbers.clear();
    _temperature = 273.15f;
    _sigma       = 0.0f;
    _seed        = atlas::DEFAULT_UNSIGNED_INT_SEED;

    return generator;
}

atlas::host_shared_ptr<MaxwellSigmaGenerator>
MaxwellSigmaGenerator::Builder::make_host_shared() {
    return atlas::make_host_shared<MaxwellSigmaGenerator>(build());
}

void
MaxwellSigmaGenerator::Builder::validate() const {
    if (_species_ratios.empty() || _species_numbers.empty()) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: species ratios/numbers must not be empty.");
    }

    if (_species_ratios.size() != _species_numbers.size()) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: species ratios/numbers size mismatch.");
    }

    if (!atlas::isfinite(_temperature) || _temperature < 0.0f) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: temperature must be finite and non-negative.");
    }

    if (!atlas::isfinite(_sigma) || _sigma < 0.0f) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: sigma must be finite and non-negative.");
    }
}

}
