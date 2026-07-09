#include <atlas/generator/jittering_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace atlas {

JitteringGenerator::JitteringGenerator(DeviceBuffer<float> species_ratios,
                                       DeviceBuffer<float> species_numbers,
                                       const float temperature,
                                       const float base_value,
                                       const float jitter_radius,
                                       Float3 bulk_velocity,
                                       const unsigned int seed) noexcept
    : _species_ratios(std::move(species_ratios))
    , _species_numbers(std::move(species_numbers))
    , _temperature(temperature)
    , _base_value(base_value)
    , _jitter_radius(jitter_radius)
    , _bulk_velocity(bulk_velocity)
    , _seed(seed) {
}

JitteringGenerator::Builder
JitteringGenerator::builder() noexcept {
    return Builder {};
}

int
JitteringGenerator::generate(FluidVelocityState* velocities,
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

    const float        base_value    = _base_value;
    const float        radius        = std::abs(_jitter_radius);
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
            velocity_ptr[index]    = Float3(base_value, base_value, base_value)
                + atlas::sample_uniform_vector(engine, -radius, radius)
                + bulk;
        });

    return static_cast<int>(writable);
}

JitteringGenerator::Builder&
JitteringGenerator::Builder::with_species_ratios(const HostBuffer<float>& species_ratios) {
    _species_ratios = species_ratios;
    return *this;
}

JitteringGenerator::Builder&
JitteringGenerator::Builder::with_species_numbers(const HostBuffer<float>& species_numbers) {
    _species_numbers = species_numbers;
    return *this;
}

JitteringGenerator::Builder&
JitteringGenerator::Builder::with_temperature(const float temperature) noexcept {
    _temperature = temperature;
    return *this;
}

JitteringGenerator::Builder&
JitteringGenerator::Builder::with_base_value(const float base_value) noexcept {
    _base_value = base_value;
    return *this;
}

JitteringGenerator::Builder&
JitteringGenerator::Builder::with_jitter_radius(const float jitter_radius) noexcept {
    _jitter_radius = jitter_radius;
    return *this;
}

JitteringGenerator::Builder&
JitteringGenerator::Builder::with_bulk_velocity(const Float3& bulk_velocity) noexcept {
    _bulk_velocity = bulk_velocity;
    return *this;
}

JitteringGenerator::Builder&
JitteringGenerator::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

JitteringGenerator
JitteringGenerator::Builder::build() {
    validate();

    JitteringGenerator generator(
        DeviceBuffer<float>(_species_ratios.begin(), _species_ratios.end()),
        DeviceBuffer<float>(_species_numbers.begin(), _species_numbers.end()),
        _temperature,
        _base_value,
        _jitter_radius,
        _bulk_velocity,
        _seed);

    _species_ratios.clear();
    _species_numbers.clear();
    _temperature   = 273.15f;
    _base_value    = 0.0f;
    _jitter_radius = 0.0f;
    _bulk_velocity = Float3(0.0f, 0.0f, 0.0f);
    _seed          = atlas::DEFAULT_UNSIGNED_INT_SEED;

    return generator;
}

atlas::host_shared_ptr<JitteringGenerator>
JitteringGenerator::Builder::make_host_shared() {
    return atlas::make_host_shared<JitteringGenerator>(build());
}

void
JitteringGenerator::Builder::validate() const {
    if (_species_ratios.empty() || _species_numbers.empty()) {
        throw std::runtime_error("JitteringGenerator::Builder: species ratios/numbers must not be empty.");
    }

    if (_species_ratios.size() != _species_numbers.size()) {
        throw std::runtime_error("JitteringGenerator::Builder: species ratios/numbers size mismatch.");
    }

    if (!atlas::isfinite(_temperature) || _temperature < 0.0f) {
        throw std::runtime_error("JitteringGenerator::Builder: temperature must be finite and non-negative.");
    }

    if (!atlas::isfinite(_base_value) || !atlas::isfinite(_jitter_radius)) {
        throw std::runtime_error("JitteringGenerator::Builder: base value and jitter radius must be finite.");
    }
}

}
