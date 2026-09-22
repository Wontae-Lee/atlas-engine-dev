#include "rendering/state/raw_state_provider.h"

#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/system/system.h>

#include <optional>
#include <stdexcept>

namespace atlas::interactive {

namespace {

template <typename State>
void
upload_required(const Fluid& fluid,
                const std::size_t count,
                StateBridge& bridge,
                opengl::Buffer& target) {
    const State* state = fluid.state<State>();
    if (state == nullptr) throw std::logic_error("Fluid is missing a required particle state.");
    bridge.upload(state->data(), count, target);
}

template <typename State>
void
upload_optional(const Fluid& fluid,
                const std::size_t count,
                StateBridge& bridge,
                std::optional<opengl::Buffer>& target) {
    const State* state = fluid.state<State>();
    if (state == nullptr) {
        if (target) bridge.release(*target);
        target.reset();
        return;
    }
    if (!target) target.emplace();
    bridge.upload(state->data(), count, *target);
}

}

const RenderState&
RawStateProvider::update(const System& system) {
    if (!system.fluid()) throw std::logic_error("RawStateProvider requires a System with a Fluid.");

    const Fluid& fluid = *system.fluid();
    const std::size_t count = fluid.particle_count();
    _state.particle_count = count;

    upload_required<FluidPositionState>(fluid, count, _bridge, _state.position);
    upload_required<FluidVelocityState>(fluid, count, _bridge, _state.velocity);
    upload_required<FluidSpeciesState>(fluid, count, _bridge, _state.species);
    upload_optional<FluidTemperatureState>(fluid, count, _bridge, _state.temperature);
    upload_optional<FluidTranslationalEnergyState>(fluid,
                                                   count,
                                                   _bridge,
                                                   _state.translational_energy);
    upload_optional<FluidRotationalEnergyState>(fluid,
                                                count,
                                                _bridge,
                                                _state.rotational_energy);
    upload_optional<FluidVibrationalEnergyState>(fluid,
                                                 count,
                                                 _bridge,
                                                 _state.vibrational_energy);
    return _state;
}

}
