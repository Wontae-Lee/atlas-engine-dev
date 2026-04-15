#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>

#include <cstddef>
#include <utility>

namespace atlas::system {

struct FluidState {
    FluidState() = default;

    FluidState(const FluidState&) = delete;

    FluidState(FluidState&&) noexcept = default;

    virtual ~FluidState() = default;

    FluidState&
    operator=(const FluidState&)
        = delete;

    FluidState&
    operator=(FluidState&&) noexcept
        = default;
};

template <typename T>
struct ParticleState final : FluidState {
    ParticleState() = default;

    explicit ParticleState(const std::size_t buffer_size)
        : position(buffer_size)
        , velocity(buffer_size)
        , species(buffer_size)
        , active(buffer_size) { }

    ParticleState(DeviceBuffer<Vector3<T>> position_,
                  DeviceBuffer<Vector3<T>> velocity_,
                  DeviceBuffer<std::size_t> species_,
                  DeviceBuffer<int> active_) noexcept
        : position(std::move(position_))
        , velocity(std::move(velocity_))
        , species(std::move(species_))
        , active(std::move(active_)) { }

    DeviceBuffer<Vector3<T>> position;
    DeviceBuffer<Vector3<T>> velocity;
    DeviceBuffer<std::size_t> species;
    DeviceBuffer<int> active;

};

template <typename T>
struct FluidTemperatureState final : FluidState {
    FluidTemperatureState() = default;

    explicit FluidTemperatureState(const std::size_t buffer_size)
        : temperature(buffer_size) { }

    explicit FluidTemperatureState(DeviceBuffer<T> temperature_) noexcept
        : temperature(std::move(temperature_)) { }

    DeviceBuffer<T> temperature;
};

}

namespace atlas {

using FluidState = atlas::system::FluidState;

template <typename T>
using ParticleState = atlas::system::ParticleState<T>;

template <typename T>
using FluidTemperatureState = atlas::system::FluidTemperatureState<T>;

}
