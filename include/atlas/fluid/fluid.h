#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/material/material_properties.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace atlas {

using FluidStateStore = TypeStore<FluidState>;

class Fluid final {
public:
    class Builder;

    Fluid() = default;

    ATLAS_HOST explicit Fluid(std::size_t buffer_size);

    Fluid(const Fluid&) = delete;

    Fluid(Fluid&&) noexcept = default;

    ~Fluid() = default;

    Fluid&
    operator=(const Fluid&)
        = delete;

    Fluid&
    operator=(Fluid&&) noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Generate>&
    generators() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Generate>&
    generators() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<MaterialProperties>&
    particle_properties() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<MaterialProperties>&
    particle_properties() noexcept;

    ATLAS_HOST void
    set_particle_count(std::size_t particle_count);

    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args) {
        return _states.template emplace<StateT>(std::forward<Args>(args)...);
    }

    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state) {
        _states.template set<StateT>(std::move(state));
    }

    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE StateT*
    state() noexcept {
        return _states.template get<StateT>();
    }

    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const StateT*
    state() const noexcept {
        return _states.template get<StateT>();
    }

    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE bool
    has_state() const noexcept {
        return _states.template contains<StateT>();
    }

    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state() {
        return _states.template remove<StateT>();
    }

    ATLAS_NODISCARD ATLAS_HOST FluidStateStore&
    states() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const FluidStateStore&
    states() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    buffer_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    particle_count() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    statistical_weight() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const ObserverHostPtr&
    observer() const noexcept;

    ATLAS_HOST void
    save(std::string_view path) const;

private:
    friend class Builder;

    DeviceBuffer<MaterialProperties> _particle_properties;

    DeviceBuffer<Generate> _generators;

    std::size_t _particle_count = 0;

    std::size_t _buffer_size = 0;

    float _statistical_weight = 1.0f;

    ObserverHostPtr _observer {};

    FluidStateStore _states;
};

class Fluid::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Fluid
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Fluid>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_properties(const HostBuffer<MaterialProperties>& properties);

    ATLAS_HOST Builder&
    with_generators(const HostBuffer<GeneratorHostPtr>& generators);

    ATLAS_HOST Builder&
    with_buffer_size(std::size_t buffer_size) noexcept;

    ATLAS_HOST Builder&
    with_statistical_weight(float statistical_weight) noexcept;

    ATLAS_HOST Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    ATLAS_HOST Builder&
    with_binary(const std::string& path);

private:
    ATLAS_HOST void
    validate() const;

private:
    HostBuffer<MaterialProperties> _particles;

    HostBuffer<Generate> _generators;

    std::size_t _buffer_size = 0;

    float _statistical_weight = 1.0f;

    ObserverHostPtr _observer {};

    std::optional<HostBuffer<Float3>> _position_state;

    std::optional<HostBuffer<Float3>> _velocity_state;

    std::optional<HostBuffer<std::size_t>> _species_state;

    std::optional<HostBuffer<int>> _active_state;

    std::optional<HostBuffer<float>> _temperature_state;

    std::optional<std::size_t> _particle_count;
};

using FluidHostPtr = atlas::host_shared_ptr<Fluid>;

using FluidDevicePtr = atlas::device_shared_ptr<Fluid>;

}
