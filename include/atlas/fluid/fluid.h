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

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace atlas {

using FluidStateStore = TypeStore<FluidState>;

template <typename T>
class Fluid final {
public:
    class Builder;

    Fluid() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Fluid(size_t buffer_size);

    Fluid(const Fluid&) = delete;

    Fluid(Fluid&&) noexcept = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ~Fluid() = default;

    Fluid&
    operator=(const Fluid&)
        = delete;

    Fluid&
    operator=(Fluid&&) noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<GenerateOperator<T>>&
    generators() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<GenerateOperator<T>>&
    generators() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<MaterialProperties<T>>&
    particle_properties() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<MaterialProperties<T>>&
    particle_properties() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_particle_count(size_t particle_count);

    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args);

    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state);

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE StateT*
    state() noexcept;

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const StateT*
    state() const noexcept;

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_state() const noexcept;

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state();

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidStateStore&
    states() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidStateStore&
    states() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    buffer_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    particle_count() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    statistical_weight() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const ObserverHostPtr&
    observer() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    save(std::string_view path) const;

private:
    friend class Builder;

    DeviceBuffer<MaterialProperties<T>> _particle_properties;

    DeviceBuffer<GenerateOperator<T>> _generators;

    size_t _particle_count = 0;

    size_t _buffer_size = 0;

    T _statistical_weight = T(1);

    ObserverHostPtr _observer {};

    FluidStateStore _states;
};

template <typename T>
class Fluid<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Fluid<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Fluid<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_properties(const HostBuffer<MaterialProperties<T>>& properties);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_generators(const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_buffer_size(size_t buffer_size) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_statistical_weight(T statistical_weight) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_binary(const std::string& path);

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    HostBuffer<MaterialProperties<T>> _particles;

    HostBuffer<GenerateOperator<T>> _generators;

    size_t _buffer_size = 0;

    T _statistical_weight = T(1.0);

    ObserverHostPtr _observer {};

    std::optional<HostBuffer<Vector3<T>>> _position_state;

    std::optional<HostBuffer<Vector3<T>>> _velocity_state;

    std::optional<HostBuffer<std::size_t>> _species_state;

    std::optional<HostBuffer<int>> _active_state;

    std::optional<HostBuffer<T>> _temperature_state;

    std::optional<std::size_t> _particle_count;
};

}

namespace atlas {

template <typename T>
using FluidHostPtr = host_shared_ptr<Fluid<T>>;

template <typename T>
using FluidDevicePtr = device_shared_ptr<Fluid<T>>;

}

#include <atlas/fluid/fluid.hpp>