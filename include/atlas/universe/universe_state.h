#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/math/vector/vector.h>

#include <cstddef>
#include <typeindex>
#include <utility>

namespace atlas::universe {

using TypeId = std::type_index;

struct UniverseState {
    UniverseState() = default;

    UniverseState(const UniverseState&) = delete;

    UniverseState(UniverseState&&) noexcept = default;

    virtual ~UniverseState() = default;

    UniverseState&
    operator=(const UniverseState&)
        = delete;

    UniverseState&
    operator=(UniverseState&&) noexcept
        = default;
};

template <typename T>
struct UniverseTemperatureState final : UniverseState {
    UniverseTemperatureState() = default;

    explicit UniverseTemperatureState(const std::size_t number_of_cells)
        : temperature(number_of_cells) { }

    explicit UniverseTemperatureState(DeviceBuffer<T> temperature) noexcept
        : temperature(std::move(temperature)) { }

    DeviceBuffer<T> temperature;
};

template <typename T>
struct UniverseBulkVelocityState final : UniverseState {
    UniverseBulkVelocityState() = default;

    explicit UniverseBulkVelocityState(const std::size_t number_of_cells)
        : bulk_velocity(number_of_cells) { }

    explicit UniverseBulkVelocityState(DeviceBuffer<Vector3<T>> bulk_velocity_) noexcept
        : bulk_velocity(std::move(bulk_velocity_)) { }

    DeviceBuffer<Vector3<T>> bulk_velocity;
};

template <typename T>
struct UniverseMomentumWeightState final : UniverseState {
    UniverseMomentumWeightState() = default;

    explicit UniverseMomentumWeightState(const std::size_t number_of_cells)
        : momentum_weight(number_of_cells) { }

    explicit UniverseMomentumWeightState(DeviceBuffer<T> momentum_weight_) noexcept
        : momentum_weight(std::move(momentum_weight_)) { }

    DeviceBuffer<T> momentum_weight;
};

template <typename T>
struct UniverseThermalEnergyState final : UniverseState {
    UniverseThermalEnergyState() = default;

    explicit UniverseThermalEnergyState(const std::size_t number_of_cells)
        : thermal_energy(number_of_cells) { }

    explicit UniverseThermalEnergyState(DeviceBuffer<T> thermal_energy_) noexcept
        : thermal_energy(std::move(thermal_energy_)) { }

    DeviceBuffer<T> thermal_energy;
};

template <typename T, std::size_t N>
struct UniverseMaterialRatioState final : UniverseState {
    static_assert(N >= 1, "UniverseMaterialRatioState dimension must be >= 1.");

    UniverseMaterialRatioState() = default;

    explicit UniverseMaterialRatioState(const std::size_t number_of_cells)
        : material_ratio(number_of_cells) { }

    explicit UniverseMaterialRatioState(DeviceBuffer<Vector<T, N>> material_ratio_) noexcept
        : material_ratio(std::move(material_ratio_)) { }

    DeviceBuffer<Vector<T, N>> material_ratio;
};

}
