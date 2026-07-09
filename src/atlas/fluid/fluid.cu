#include <atlas/fluid/fluid.h>

#include <stdexcept>
#include <utility>

namespace atlas {

Fluid::Fluid(const std::size_t buffer_size)
    : _buffer_size(buffer_size) {
    _states.reserve(4);
    emplace_state<FluidPositionState>(buffer_size);
    emplace_state<FluidVelocityState>(buffer_size);
    emplace_state<FluidSpeciesState>(buffer_size);
    emplace_state<FluidActiveState>(buffer_size);
}

Fluid::Builder
Fluid::builder() noexcept {
    return Builder {};
}

void
Fluid::set_particle_count(const std::size_t particle_count) {
    if (particle_count > _buffer_size) {
        throw std::out_of_range("Fluid::set_particle_count: particle_count exceeds buffer_size.");
    }
    _particle_count = particle_count;
}

FluidStateStore&
Fluid::states() noexcept {
    return _states;
}

const FluidStateStore&
Fluid::states() const noexcept {
    return _states;
}

std::size_t
Fluid::buffer_size() const noexcept {
    return _buffer_size;
}

std::size_t
Fluid::particle_count() const noexcept {
    return _particle_count;
}

float
Fluid::statistical_weight() const noexcept {
    return _statistical_weight;
}

Fluid::Builder&
Fluid::Builder::with_buffer_size(const std::size_t buffer_size) noexcept {
    _buffer_size = buffer_size;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_particle_count(const std::size_t particle_count) noexcept {
    _particle_count = particle_count;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_statistical_weight(const float statistical_weight) noexcept {
    _statistical_weight = statistical_weight;
    return *this;
}

void
Fluid::Builder::validate() const {
    if (!(_statistical_weight > 0.0f)) {
        throw std::runtime_error(
            "Fluid::Builder: statistical_weight must be positive.");
    }
    if (_particle_count > _buffer_size) {
        throw std::runtime_error(
            "Fluid::Builder: particle_count exceeds buffer_size.");
    }
}

Fluid
Fluid::Builder::build() const {
    validate();

    Fluid fluid(_buffer_size);
    fluid._statistical_weight = _statistical_weight;
    fluid.set_particle_count(_particle_count);

    return fluid;
}

atlas::host_shared_ptr<Fluid>
Fluid::Builder::make_host_shared() const {
    return atlas::make_host_shared<Fluid>(build());
}

}
