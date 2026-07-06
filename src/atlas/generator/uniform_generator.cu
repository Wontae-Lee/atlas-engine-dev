#include <atlas/generator/uniform_generator.h>

#include <stdexcept>

namespace atlas {

UniformGenerator::Builder
UniformGenerator::builder() noexcept {
    return Builder {};
}

UniformGenerator::UniformGenerator(const float min_value,
                                   const float max_value,
                                   const unsigned int seed) noexcept
    : _min_value(min_value)
    , _max_value(max_value)
    , _operator(UniformGenerate(seed)) {
}

Float3
UniformGenerator::generate() const {
    return _operator.generate(_min_value, _max_value);
}

const Generate&
UniformGenerator::generate_operator() const noexcept {
    return _operator;
}

Generate
UniformGenerator::make_generate_operator() const noexcept {
    return _operator;
}

float
UniformGenerator::param0() const noexcept {
    return _min_value;
}

float
UniformGenerator::param1() const noexcept {
    return _max_value;
}

GenerateType
UniformGenerator::type() const noexcept {
    return GenerateType::uniform;
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
UniformGenerator::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

UniformGenerator
UniformGenerator::Builder::build() const {
    validate();
    return UniformGenerator(*_min_value, *_max_value, _seed);
}

atlas::host_shared_ptr<UniformGenerator>
UniformGenerator::Builder::make_host_shared() const {
    return atlas::make_host_shared<UniformGenerator>(build());
}

void
UniformGenerator::Builder::validate() const {
    if (!_min_value.has_value() || !_max_value.has_value()) {
        throw std::runtime_error(
            "UniformGenerator::Builder: min_value and max_value must be provided.");
    }

    if (!(*_min_value < *_max_value)) {
        throw std::runtime_error(
            "UniformGenerator::Builder: min_value must be less than max_value.");
    }
}

}
