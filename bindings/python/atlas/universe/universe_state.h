#pragma once

#include "../detail/array.h"

#include <atlas/universe/universe.h>

#include <nanobind/nanobind.h>

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

namespace atlas::python {

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

inline nanobind::object
read_universe_state(const Universe& universe, const std::string& name) {
    const std::size_t count = static_cast<std::size_t>(universe.cell_count());
    return visit_universe_state(name, [&]<typename State>() -> nanobind::object {
        const auto* state = universe.state<State>();
        return state ? numpy_copy_device(state->data(), count) : nanobind::none();
    });
}

inline void
write_universe_state(Universe& universe,
                     const std::string& name,
                     nanobind::handle values,
                     const std::size_t offset = 0) {
    visit_universe_state(name, [&]<typename State>() {
        using Buffer = std::remove_reference_t<decltype(std::declval<State&>().data())>;
        using T      = typename Buffer::value_type;
        const std::size_t count = numpy_size<T>(values);
        const std::size_t capacity = static_cast<std::size_t>(universe.cell_count());
        if (offset > capacity || count > capacity - offset) {
            throw nanobind::value_error("array exceeds the state buffer capacity");
        }
        auto* state = universe.state<State>();
        if (!state) state = &universe.emplace_state<State>(capacity);
        write_array(state->data(), values, offset);
    });
}

inline bool
has_universe_state(const Universe& universe, const std::string& name) {
    return visit_universe_state(name, [&]<typename State>() { return universe.has_state<State>(); });
}

inline bool
remove_universe_state(Universe& universe, const std::string& name) {
    return visit_universe_state(name, [&]<typename State>() {
        return static_cast<bool>(universe.remove_state<State>());
    });
}

inline void
reset_universe_state(Universe& universe, const std::string& name) {
    visit_universe_state(name, [&]<typename State>() {
        auto* state = universe.state<State>();
        if (!state) throw nanobind::key_error("state is not attached");
        state->reset();
    });
}

}
