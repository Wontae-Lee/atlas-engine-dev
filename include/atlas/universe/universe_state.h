#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/math/vector/vector.h>

#include <cstddef>
#include <typeindex>
#include <utility>

namespace atlas::universe {

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
struct UniverseTemperature final : UniverseState {
    UniverseTemperature() = default;

    explicit UniverseTemperature(const std::size_t number_of_cells)
        : temperature(number_of_cells) { }

    explicit UniverseTemperature(DeviceBuffer<T> temperature) noexcept
        : temperature(std::move(temperature)) { }

    DeviceBuffer<T> temperature;
};

template <typename T, std::size_t N>
struct UniverseMaterialRatio final : UniverseState {
    static_assert(N >= 1, "UniverseMaterialRatio dimension must be >= 1.");

    UniverseMaterialRatio() = default;

    explicit UniverseMaterialRatio(const std::size_t number_of_cells)
        : material_ratio(number_of_cells) { }

    explicit UniverseMaterialRatio(DeviceBuffer<Vector<T, N>> material_ratio_) noexcept
        : material_ratio(std::move(material_ratio_)) { }

    DeviceBuffer<Vector<T, N>> material_ratio;
};

}

namespace atlas {

using TypeId = std::type_index;

using State = atlas::universe::UniverseState;

template <typename T>
using Temperature = atlas::universe::UniverseTemperature<T>;

template <typename T, std::size_t N>
using MaterialRatio = atlas::universe::UniverseMaterialRatio<T, N>;

}
