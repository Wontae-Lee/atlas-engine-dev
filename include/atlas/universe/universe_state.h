#pragma once

#include <typeindex>

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

    explicit UniverseTemperature(DeviceBuffer<T> values) noexcept
        : values(std::move(values)) { }

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<T>&
    data() noexcept {

        return values;
    }

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<T>&
    data() const noexcept {

        return values;
    }

    DeviceBuffer<T> values;
};

}

namespace atlas {

using TypeId = std::type_index;

using State = atlas::universe::UniverseState;

template <typename T>
using Temperature = atlas::universe::UniverseTemperature<T>;

}