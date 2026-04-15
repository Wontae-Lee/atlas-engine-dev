#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/material/matrial_properties.h>
#include <atlas/memory/memory.h>

#include <memory>
#include <typeindex>
#include <unordered_map>

namespace atlas::system {

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
    operator=(Fluid&&) noexcept
        = default;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<GenerateOperator<T>>&
    generators() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<GenerateOperator<T>>&
    generators() noexcept;

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

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    buffer_size() const noexcept;

private:
    friend class Builder;

private:
    DeviceBuffer<MatrialProperties<T>> _particle_properties;
    DeviceBuffer<GenerateOperator<T>> _generators;

    size_t _buffer_size = 0;

    std::unordered_map<std::type_index, std::unique_ptr<FluidState>> _states;
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
    with_properties(const HostBuffer<MatrialProperties<T>>& properties);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_generators(const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_buffer_size(size_t buffer_size) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    DeviceBuffer<MatrialProperties<T>> _particles;
    DeviceBuffer<GenerateOperator<T>> _generators;

    size_t _buffer_size = 0;
};

}

namespace atlas {

template <typename T>
using Fluid = system::Fluid<T>;

template <typename T>
using FluidHostPtr = atlas::host_shared_ptr<system::Fluid<T>>;

}

#include <atlas/fluid/fluid.hpp>
