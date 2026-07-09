#include <atlas/generator/uniform_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace atlas {

UniformGenerator::UniformGenerator(DeviceBuffer<float> species_ratios,
                                   DeviceBuffer<float> species_numbers,
                                   const float temperature,
                                   const float min_value,
                                   const float max_value,
                                   Float3 bulk_velocity,
                                   const unsigned int seed) noexcept
    : _species_ratios(std::move(species_ratios))
    , _species_numbers(std::move(species_numbers))
    , _temperature(temperature)
    , _min_value(min_value)
    , _max_value(max_value)
    , _bulk_velocity(bulk_velocity)
    , _seed(seed) {
}

UniformGenerator::Builder
UniformGenerator::builder() noexcept {
    return Builder {};
}

int
UniformGenerator::generate(FluidVelocityState* velocities,
                           FluidSpeciesState* species,
                           const std::size_t offset,
                           const std::size_t count) const {
    if (velocities == nullptr || species == nullptr || count == 0) {
        return 0;
    }

    DeviceBuffer<Float3>&      velocity_buffer = velocities->data();
    DeviceBuffer<std::size_t>& species_buffer  = species->data();

    const std::size_t capacity = std::min(velocity_buffer.size(), species_buffer.size());
    if (offset >= capacity) {
        return 0;
    }

    const std::size_t writable = std::min(count, capacity - offset);
    if (writable == 0) {
        return 0;
    }

    const float        min_value     = _min_value;
    const float        max_value     = _max_value;
    const Float3       bulk          = _bulk_velocity;
    const unsigned int seed          = _seed;
    const int          species_count = static_cast<int>(_species_ratios.size());

    const float* ratios          = atlas::raw_pointer_cast(_species_ratios.data());
    const float* numbers         = atlas::raw_pointer_cast(_species_numbers.data());
    Float3*      velocity_ptr    = atlas::raw_pointer_cast(velocity_buffer.data());
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
            velocity_ptr[index]    = atlas::sample_uniform_vector(engine, min_value, max_value) + bulk;
        });

    return static_cast<int>(writable);
}

UniformGenerator::Builder&
UniformGenerator::Builder::with_species_ratios(const HostBuffer<float>& species_ratios) {
    _species_ratios = species_ratios;
    return *this;
}

UniformGenerator::Builder&
UniformGenerator::Builder::with_species_numbers(const HostBuffer<float>& species_numbers) {
    _species_numbers = species_numbers;
    return *this;
}

UniformGenerator::Builder&
UniformGenerator::Builder::with_temperature(const float temperature) noexcept {
    _temperature = temperature;
    return *this;
}

UniformGenerator::Builder&
UniformGenerator::Builder::with_min_value(const float min_value) noexcept {
    _min_value = min_value;
    return *this;
}

UniformGenerator::Builder&
UniformGenerator::Builder::with_max_value(const float max_value) noexcept {
    _max_value = max_value;
    return *this;
}

UniformGenerator::Builder&
UniformGenerator::Builder::with_bulk_velocity(const Float3& bulk_velocity) noexcept {
    _bulk_velocity = bulk_velocity;
    return *this;
}

UniformGenerator::Builder&
UniformGenerator::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

UniformGenerator
UniformGenerator::Builder::build() {
    validate();

    UniformGenerator generator(
        DeviceBuffer<float>(_species_ratios.begin(), _species_ratios.end()),
        DeviceBuffer<float>(_species_numbers.begin(), _species_numbers.end()),
        _temperature,
        _min_value,
        _max_value,
        _bulk_velocity,
        _seed);

    _species_ratios.clear();
    _species_numbers.clear();
    _temperature = 273.15f;
    _min_value   = 0.0f;
    _max_value     = 0.0f;
    _bulk_velocity = Float3(0.0f, 0.0f, 0.0f);
    _seed          = atlas::DEFAULT_UNSIGNED_INT_SEED;

    return generator;
}

atlas::host_shared_ptr<UniformGenerator>
UniformGenerator::Builder::make_host_shared() {
    return atlas::make_host_shared<UniformGenerator>(build());
}

void
UniformGenerator::Builder::validate() const {
    if (_species_ratios.empty() || _species_numbers.empty()) {
        throw std::runtime_error("UniformGenerator::Builder: species ratios/numbers must not be empty.");
    }

    if (_species_ratios.size() != _species_numbers.size()) {
        throw std::runtime_error("UniformGenerator::Builder: species ratios/numbers size mismatch.");
    }

    if (!atlas::isfinite(_temperature) || _temperature < 0.0f) {
        throw std::runtime_error("UniformGenerator::Builder: temperature must be finite and non-negative.");
    }

    if (!atlas::isfinite(_min_value) || !atlas::isfinite(_max_value) || _min_value > _max_value) {
        throw std::runtime_error("UniformGenerator::Builder: require finite min_value <= max_value.");
    }
}

}
