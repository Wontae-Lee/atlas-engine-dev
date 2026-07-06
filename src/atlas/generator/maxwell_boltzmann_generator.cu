#include <atlas/generator/maxwell_boltzmann_generator.h>

#include <stdexcept>

namespace atlas {

MaxwellBoltzmannGenerator::Builder
MaxwellBoltzmannGenerator::builder() noexcept {
    return Builder {};
}

MaxwellBoltzmannGenerator::MaxwellBoltzmannGenerator(const float temperature,
                                                     const float molecular_mass,
                                                     const Float3& bulk_velocity,
                                                     const unsigned int seed) noexcept
    : _temperature(temperature)
    , _molecular_mass(molecular_mass)
    , _bulk_velocity(bulk_velocity)
    , _operator(MaxwellBoltzmannGenerate(seed, bulk_velocity)) {
}

Float3
MaxwellBoltzmannGenerator::generate() const {
    return _operator.generate(_temperature, _molecular_mass);
}

const Generate&
MaxwellBoltzmannGenerator::generate_operator() const noexcept {
    return _operator;
}

Generate
MaxwellBoltzmannGenerator::make_generate_operator() const noexcept {
    return _operator;
}

float
MaxwellBoltzmannGenerator::param0() const noexcept {
    return _temperature;
}

float
MaxwellBoltzmannGenerator::param1() const noexcept {
    return _molecular_mass;
}

GenerateType
MaxwellBoltzmannGenerator::type() const noexcept {
    return GenerateType::maxwell_boltzmann;
}

MaxwellBoltzmannGenerator::Builder&
MaxwellBoltzmannGenerator::Builder::with_temperature(const float temperature) noexcept {
    _temperature = temperature;
    return *this;
}

MaxwellBoltzmannGenerator::Builder&
MaxwellBoltzmannGenerator::Builder::with_molecular_mass(const float molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
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

MaxwellBoltzmannGenerator
MaxwellBoltzmannGenerator::Builder::build() const {
    validate();
    return MaxwellBoltzmannGenerator(
        *_temperature,
        *_molecular_mass,
        _bulk_velocity,
        _seed);
}

atlas::host_shared_ptr<MaxwellBoltzmannGenerator>
MaxwellBoltzmannGenerator::Builder::make_host_shared() const {
    return atlas::make_host_shared<MaxwellBoltzmannGenerator>(build());
}

void
MaxwellBoltzmannGenerator::Builder::validate() const {
    if (!_temperature.has_value() || !_molecular_mass.has_value()) {
        throw std::runtime_error(
            "MaxwellBoltzmannGenerator::Builder: temperature and molecular_mass must be provided.");
    }

    if (!(*_temperature > 0.0f)) {
        throw std::runtime_error(
            "MaxwellBoltzmannGenerator::Builder: temperature must be greater than zero.");
    }

    if (!(*_molecular_mass > 0.0f)) {
        throw std::runtime_error(
            "MaxwellBoltzmannGenerator::Builder: molecular_mass must be greater than zero.");
    }
}

}
