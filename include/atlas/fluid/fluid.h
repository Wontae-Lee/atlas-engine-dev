#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/memory/memory.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace atlas {

using FluidStateStore = TypeStore<FluidState>;

// Owns the particle buffers (as a store of FluidState leaves), the survival
// flag that drives compaction, and the two counters describing how much of the
// buffers is live. Materials, generators and observers are held by whoever
// drives the simulation, not by the fluid.
class Fluid final {
public:
    class Builder;

    Fluid() = default;

    ATLAS_HOST explicit Fluid(std::size_t buffer_size);

    ATLAS_HOST
    Fluid(std::size_t buffer_size, MaterialDictionaryHostPtr materials);

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

    ATLAS_HOST void
    set_particle_count(std::size_t particle_count);

    // One entry per particle: non-zero keeps it, zero drops it at the next
    // compact(). Whoever despawns particles writes here.
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    active() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    active() const noexcept;

    // Drops every particle the active flags mark dead, gathering the survivors
    // to the front of every state and shrinking particle_count to match.
    // Returns how many survived; a fluid whose particles all survive is left
    // untouched.
    ATLAS_HOST std::size_t
    compact();

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

    // The raw-pointer face of a set of states, for a kernel to capture by
    // value. See fluid_view.h; ViewT only has to expose a static make(Fluid&).
    template <typename ViewT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE ViewT
    view() {
        return ViewT::make(*this);
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

    // The per-species properties, indexed by FluidSpeciesState.
    ATLAS_NODISCARD ATLAS_HOST const MaterialDictionaryHostPtr&
    materials() const noexcept;

private:
    friend class Builder;

    std::size_t _particle_count = 0;

    std::size_t _buffer_size = 0;

    float _statistical_weight = 1.0f;

    MaterialDictionaryHostPtr _materials {};

    FluidStateStore _states;

    DeviceBuffer<int> _active;

    // Scratch for compact(): the exclusive prefix sum of the survival flags,
    // the gather indices it produces, and a one-element device total.
    DeviceBuffer<int> _survivor_offsets;

    DeviceBuffer<std::size_t> _compact_indices;

    DeviceBuffer<int> _survivor_total;
};

class Fluid::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_buffer_size(std::size_t buffer_size) noexcept;

    ATLAS_HOST Builder&
    with_particle_count(std::size_t particle_count) noexcept;

    ATLAS_HOST Builder&
    with_statistical_weight(float statistical_weight) noexcept;

    ATLAS_HOST Builder&
    with_materials(MaterialDictionaryHostPtr materials) noexcept;

    ATLAS_NODISCARD ATLAS_HOST Fluid
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Fluid>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    std::size_t _particle_count = 0;

    std::size_t _buffer_size = 0;

    float _statistical_weight = 1.0f;

    MaterialDictionaryHostPtr _materials {};
};

using FluidHostPtr = atlas::host_shared_ptr<Fluid>;

using FluidDevicePtr = atlas::device_shared_ptr<Fluid>;

}
