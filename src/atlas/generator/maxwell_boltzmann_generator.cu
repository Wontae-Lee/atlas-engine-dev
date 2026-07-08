#include <atlas/generator/maxwell_boltzmann_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace atlas {

MaxwellBoltzmannGenerator::MaxwellBoltzmannGenerator(DeviceBuffer<float> species_ratios,
                                                     DeviceBuffer<float> species_numbers,
                                                     DeviceBuffer<float> species_mass,
                                                     const float temperature,
                                                     Float3 bulk_velocity,
                                                     const unsigned int seed) noexcept
    : _species_ratios(std::move(species_ratios))
    , _species_numbers(std::move(species_numbers))
    , _species_mass(std::move(species_mass))
    , _temperature(temperature)
    , _bulk_velocity(bulk_velocity)
    , _seed(seed) {
}

MaxwellBoltzmannGenerator::Builder
MaxwellBoltzmannGenerator::builder() noexcept {
    return Builder {};
}

int
MaxwellBoltzmannGenerator::generate(FluidVelocityState* velocities,
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
    const Float3       bulk          = _bulk_velocity;
    const unsigned int seed          = _seed;
    const int          species_count = static_cast<int>(_species_ratios.size());

    const float* ratios          = atlas::raw_pointer_cast(_species_ratios.data());
    const float* numbers         = atlas::raw_pointer_cast(_species_numbers.data());
    const float* masses          = atlas::raw_pointer_cast(_species_mass.data());
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

            const int selected = atlas::sample_weighted_index(ratios, species_count, engine);

            species_ptr[index]     = static_cast<std::size_t>(numbers[selected]);
            temperature_ptr[index] = temperature;

            const float molecular_mass = masses[selected];

            if (temperature > 0.0f && molecular_mass > 0.0f) {
                const float sigma = atlas::sqrt_nonnegative(
                    atlas::boltzmann_constant * temperature / molecular_mass);
                velocity_ptr[index] = atlas::sample_normal_vector(engine, sigma) + bulk;
            } else {
                velocity_ptr[index] = bulk;
            }
        });

    return static_cast<int>(writable);
}

MaxwellBoltzmannGenerator::Builder&
MaxwellBoltzmannGenerator::Builder::with_species_ratios(const HostBuffer<float>& species_ratios) {
    _species_ratios = species_ratios;
    return *this;
}

MaxwellBoltzmannGenerator::Builder&
MaxwellBoltzmannGenerator::Builder::with_species_numbers(const HostBuffer<float>& species_numbers) {
    _species_numbers = species_numbers;
    return *this;
}

MaxwellBoltzmannGenerator::Builder&
MaxwellBoltzmannGenerator::Builder::with_material_dictionary(MaterialDictionaryHostPtr material_dictionary) noexcept {
    _material_dictionary = std::move(material_dictionary);
    return *this;
}

MaxwellBoltzmannGenerator::Builder&
MaxwellBoltzmannGenerator::Builder::with_species_mass(const HostBuffer<float>& species_mass) {
    _species_mass = species_mass;
    return *this;
}

MaxwellBoltzmannGenerator::Builder&
MaxwellBoltzmannGenerator::Builder::with_temperature(const float temperature) noexcept {
    _temperature = temperature;
    return *this;
}

MaxwellBoltzmannGenerator::Builder&
MaxwellBoltzmannGenerator::Builder::with_bulk_velocity(const Float3& bulk_velocity) noexcept {
    _bulk_velocity = bulk_velocity;
    return *this;
}

MaxwellBoltzmannGenerator::Builder&
MaxwellBoltzmannGenerator::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

HostBuffer<float>
MaxwellBoltzmannGenerator::Builder::resolve_species_mass() const {
    if (!_species_mass.empty()) {
        return _species_mass;
    }

    const DeviceBuffer<Material>& materials = _material_dictionary->materials();

    HostBuffer<float> masses(_species_numbers.size());
    for (std::size_t k = 0; k < _species_numbers.size(); ++k) {
        const std::size_t species_id = static_cast<std::size_t>(_species_numbers[k]);
        if (species_id >= materials.size()) {
            throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: species number out of dictionary range.");
        }
        const Material material = materials[species_id];
        masses[k]              = material.mass();
    }
    return masses;
}

MaxwellBoltzmannGenerator
MaxwellBoltzmannGenerator::Builder::build() {
    validate();

    const HostBuffer<float> species_mass = resolve_species_mass();

    MaxwellBoltzmannGenerator generator(
        DeviceBuffer<float>(_species_ratios.begin(), _species_ratios.end()),
        DeviceBuffer<float>(_species_numbers.begin(), _species_numbers.end()),
        DeviceBuffer<float>(species_mass.begin(), species_mass.end()),
        _temperature,
        _bulk_velocity,
        _seed);

    _species_ratios.clear();
    _species_numbers.clear();
    _species_mass.clear();
    _material_dictionary.reset();
    _temperature   = 273.15f;
    _bulk_velocity = Float3(0.0f, 0.0f, 0.0f);
    _seed          = atlas::DEFAULT_UNSIGNED_INT_SEED;

    return generator;
}

atlas::host_shared_ptr<MaxwellBoltzmannGenerator>
MaxwellBoltzmannGenerator::Builder::make_host_shared() {
    return atlas::make_host_shared<MaxwellBoltzmannGenerator>(build());
}

void
MaxwellBoltzmannGenerator::Builder::validate() const {
    if (_species_ratios.empty() || _species_numbers.empty()) {
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: species ratios/numbers must not be empty.");
    }

    if (_species_ratios.size() != _species_numbers.size()) {
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: species ratios/numbers size mismatch.");
    }

    if (!_species_mass.empty()) {
        if (_species_mass.size() != _species_numbers.size()) {
            throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: species mass size must match species count.");
        }
    } else if (!_material_dictionary) {
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: a material dictionary or species mass is required.");
    }

    if (!atlas::isfinite(_temperature) || _temperature < 0.0f) {
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: temperature must be finite and non-negative.");
    }
}

}
