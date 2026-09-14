#pragma once

#include "array_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/universe/universe.h>

#include <string>
#include <type_traits>
#include <utility>

namespace atlas::python {

template <typename Function>
decltype(auto)
visit_fluid_state(const std::string& name, Function&& function) {
    if (name == "position") return function.template operator()<FluidPositionState>();
    if (name == "velocity") return function.template operator()<FluidVelocityState>();
    if (name == "species") return function.template operator()<FluidSpeciesState>();
    if (name == "temperature") return function.template operator()<FluidTemperatureState>();
    if (name == "translational_energy") return function.template operator()<FluidTranslationalEnergyState>();
    if (name == "rotational_energy") return function.template operator()<FluidRotationalEnergyState>();
    if (name == "vibrational_energy") return function.template operator()<FluidVibrationalEnergyState>();
    throw nanobind::key_error("unknown fluid state");
}

template <typename Function>
decltype(auto)
visit_universe_state(const std::string& name, Function&& function) {
    if (name == "temperature") return function.template operator()<UniverseTemperatureState>();
    if (name == "bulk_velocity") return function.template operator()<UniverseBulkVelocityState>();
    if (name == "field_force") return function.template operator()<UniverseFieldForceState>();
    if (name == "gravity") return function.template operator()<UniverseGravityState>();
    if (name == "max_relative_speed") return function.template operator()<UniverseMaxRelativeSpeedState>();
    if (name == "max_sigma_g") return function.template operator()<UniverseMaxSigmaGState>();
    if (name == "thermal_energy") return function.template operator()<UniverseThermalEnergyState>();
    if (name == "number_particle") return function.template operator()<UniverseNumberParticleState>();
    if (name == "collision_count") return function.template operator()<UniverseCollisionCountState>();
    if (name == "knudsen_number") return function.template operator()<UniverseKnudsenNumberState>();
    if (name == "allocated_solver") return function.template operator()<UniverseAllocatedSolverState>();
    throw nanobind::key_error("unknown universe state");
}

template <typename Owner, typename Function>
decltype(auto)
visit_state(const std::string& name, Function&& function) {
    if constexpr (std::is_same_v<Owner, Fluid>) {
        return visit_fluid_state(name, std::forward<Function>(function));
    } else {
        return visit_universe_state(name, std::forward<Function>(function));
    }
}

template <typename Owner>
std::size_t
state_capacity(const Owner& owner) {
    if constexpr (std::is_same_v<Owner, Fluid>) return owner.buffer_size();
    else return static_cast<std::size_t>(owner.cell_count());
}

template <typename Owner>
nanobind::object
read_state(const Owner& owner, const std::string& name, const bool full = false) {
    std::size_t count = state_capacity(owner);
    if constexpr (std::is_same_v<Owner, Fluid>) {
        if (!full) count = owner.particle_count();
    }
    return visit_state<Owner>(name, [&]<typename State>() -> nanobind::object {
        const auto* state = owner.template state<State>();
        return state ? numpy_copy(state->data(), count) : nanobind::none();
    });
}

template <typename Owner>
void
write_state(Owner& owner, const std::string& name, nanobind::handle values, const std::size_t offset = 0) {
    visit_state<Owner>(name, [&]<typename State>() {
        using Buffer = std::remove_reference_t<decltype(std::declval<State&>().data())>;
        using T = typename Buffer::value_type;
        const auto host = numpy_to_host<T>(values);
        const auto capacity = state_capacity(owner);
        if (offset > capacity || host.size() > capacity - offset) {
            throw nanobind::value_error("array exceeds the state buffer capacity");
        }
        if constexpr (std::is_same_v<Owner, Fluid>) {
            if constexpr (std::is_same_v<State, FluidSpeciesState>) {
                if (owner.materials()) {
                    for (const auto species : host) {
                        if (species >= owner.materials()->size()) {
                            throw nanobind::value_error("species id is outside the material dictionary");
                        }
                    }
                }
            }
        }
        auto* state = owner.template state<State>();
        if (!state) state = &owner.template emplace_state<State>(capacity);
        write_array(state->data(), host, offset);
    });
}

template <typename Owner>
bool
has_state(const Owner& owner, const std::string& name) {
    return visit_state<Owner>(name, [&]<typename State>() { return owner.template has_state<State>(); });
}

template <typename Owner>
bool
remove_state(Owner& owner, const std::string& name) {
    return visit_state<Owner>(name, [&]<typename State>() {
        return static_cast<bool>(owner.template remove_state<State>());
    });
}

template <typename Owner>
void
reset_state(Owner& owner, const std::string& name) {
    visit_state<Owner>(name, [&]<typename State>() {
        auto* state = owner.template state<State>();
        if (!state) throw nanobind::key_error("state is not attached");
        state->reset();
    });
}

inline void
write_active(Fluid& fluid, nanobind::handle values, const std::size_t offset = 0) {
    const auto host = numpy_to_host<int>(values);
    for (const int value : host) {
        if (value != 0 && value != 1) {
            throw nanobind::value_error("active flags must be 0 or 1");
        }
    }
    write_array(fluid.active(), host, offset);
}

}
