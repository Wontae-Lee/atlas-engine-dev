#include <atlas/generator/maxwell_sigma_generator.h>

#include <stdexcept>

namespace atlas {

MaxwellSigmaGenerator::Builder
MaxwellSigmaGenerator::builder() noexcept {
    return Builder {};
}

MaxwellSigmaGenerator::MaxwellSigmaGenerator(const float sigma,
                                             const unsigned int seed) noexcept
    : _sigma(sigma)
    , _operator(MaxwellSigmaGenerate(seed)) {
}

Float3
MaxwellSigmaGenerator::generate() const {
    return _operator.generate(_sigma);
}

const Generate&
MaxwellSigmaGenerator::generate_operator() const noexcept {
    return _operator;
}

Generate
MaxwellSigmaGenerator::make_generate_operator() const noexcept {
    return _operator;
}

float
MaxwellSigmaGenerator::param0() const noexcept {
    return _sigma;
}

float
MaxwellSigmaGenerator::param1() const noexcept {
    return 1.0f;
}

GenerateType
MaxwellSigmaGenerator::type() const noexcept {
    return GenerateType::maxwell_sigma;
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
MaxwellSigmaGenerator::Builder::build() const {
    validate();
    return MaxwellSigmaGenerator(*_sigma, _seed);
}

atlas::host_shared_ptr<MaxwellSigmaGenerator>
MaxwellSigmaGenerator::Builder::make_host_shared() const {
    return atlas::make_host_shared<MaxwellSigmaGenerator>(build());
}

void
MaxwellSigmaGenerator::Builder::validate() const {
    if (!_sigma.has_value()) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: sigma must be provided.");
    }

    if (!(*_sigma > 0.0f)) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: sigma must be greater than zero.");
    }
}

}
