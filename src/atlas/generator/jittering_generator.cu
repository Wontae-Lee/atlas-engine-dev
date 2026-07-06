#include <atlas/generator/jittering_generator.h>

#include <stdexcept>

namespace atlas {

JitteringGenerator::Builder
JitteringGenerator::builder() noexcept {
    return Builder {};
}

JitteringGenerator::JitteringGenerator(const float base_value,
                                     const float jitter_radius,
                                     const unsigned int seed) noexcept
    : _base_value(base_value)
    , _jitter_radius(jitter_radius)
    , _operator(JitteringGenerate(seed, base_value, jitter_radius)) {
}

Vector3
JitteringGenerator::generate() const {
    return _operator.generate(_base_value, _jitter_radius);
}

const Generate&
JitteringGenerator::generate_operator() const noexcept {
    return _operator;
}

Generate
JitteringGenerator::make_generate_operator() const noexcept {
    return _operator;
}

float
JitteringGenerator::param0() const noexcept {
    return _base_value;
}

float
JitteringGenerator::param1() const noexcept {
    return _jitter_radius;
}

GenerateType
JitteringGenerator::type() const noexcept {
    return GenerateType::jittering;
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
JitteringGenerator::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

JitteringGenerator
JitteringGenerator::Builder::build() const {
    validate();
    return JitteringGenerator(*_base_value, *_jitter_radius, _seed);
}

atlas::host_shared_ptr<JitteringGenerator>
JitteringGenerator::Builder::make_host_shared() const {
    return atlas::make_host_shared<JitteringGenerator>(build());
}

void
JitteringGenerator::Builder::validate() const {
    if (!_base_value.has_value() || !_jitter_radius.has_value()) {
        throw std::runtime_error(
            "JitteringGenerator::Builder: base_value and jitter_radius must be provided.");
    }

    if (!atlas::isfinite(*_base_value)
        || !atlas::isfinite(*_jitter_radius)) {
        throw std::runtime_error("JitteringGenerator::Builder: parameters must be finite.");
    }

    if (!(*_jitter_radius >= 0.0f)) {
        throw std::runtime_error("JitteringGenerator::Builder: jitter_radius must be non-negative.");
    }
}

}
