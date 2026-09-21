#pragma once

#include "../detail/array.h"

#include <atlas/fluid/fluid.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
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

inline nanobind::object
read_fluid_state(const Fluid& fluid, const std::string& name, const bool full = false) {
    const std::size_t count = full ? fluid.buffer_size() : fluid.particle_count();
    return visit_fluid_state(name, [&]<typename State>() -> nanobind::object {
        const auto* state = fluid.state<State>();
        return state ? numpy_copy_device(state->data(), count) : nanobind::none();
    });
}

inline void
write_fluid_state(Fluid& fluid,
                  const std::string& name,
                  nanobind::handle values,
                  const std::size_t offset = 0) {
    visit_fluid_state(name, [&]<typename State>() {
        using Buffer = std::remove_reference_t<decltype(std::declval<State&>().data())>;
        using T      = typename Buffer::value_type;
        const std::size_t count = numpy_size<T>(values);
        const std::size_t capacity = fluid.buffer_size();
        if (offset > capacity || count > capacity - offset) {
            throw nanobind::value_error("array exceeds the state buffer capacity");
        }
        if constexpr (std::is_same_v<State, FluidSpeciesState>) {
            if (fluid.materials()) {
                using Array = nanobind::ndarray<const NumpyScalar<T>, nanobind::shape<-1>,
                                                 nanobind::c_contig, nanobind::device::cpu>;
                const auto array = nanobind::cast<Array>(values);
                for (std::size_t i = 0; i < count; ++i) {
                    if (array.data()[i] >= fluid.materials()->size()) {
                        throw nanobind::value_error("species id is outside the material dictionary");
                    }
                }
            }
        }
        auto* state = fluid.state<State>();
        if (!state) state = &fluid.emplace_state<State>(capacity);
        write_array(state->data(), values, offset);
    });
}

inline bool
has_fluid_state(const Fluid& fluid, const std::string& name) {
    return visit_fluid_state(name, [&]<typename State>() { return fluid.has_state<State>(); });
}

inline bool
remove_fluid_state(Fluid& fluid, const std::string& name) {
    return visit_fluid_state(name, [&]<typename State>() {
        return static_cast<bool>(fluid.remove_state<State>());
    });
}

inline void
reset_fluid_state(Fluid& fluid, const std::string& name) {
    visit_fluid_state(name, [&]<typename State>() {
        auto* state = fluid.state<State>();
        if (!state) throw nanobind::key_error("state is not attached");
        state->reset();
    });
}

inline void
write_active(Fluid& fluid, nanobind::handle values, const std::size_t offset = 0) {
    using Array = nanobind::ndarray<const int, nanobind::shape<-1>,
                                     nanobind::c_contig, nanobind::device::cpu>;
    const auto array = nanobind::cast<Array>(values);
    for (std::size_t i = 0; i < array.shape(0); ++i) {
        if (array.data()[i] != 0 && array.data()[i] != 1) {
            throw nanobind::value_error("active flags must be 0 or 1");
        }
    }
    write_array(fluid.active(), values, offset);
}

}
