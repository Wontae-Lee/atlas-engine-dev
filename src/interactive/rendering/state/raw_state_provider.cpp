#include "rendering/state/raw_state_provider.h"

#include "view/simulation_render_view.h"

#include <optional>
#include <stdexcept>

namespace atlas::interactive {

namespace {

void
upload_required(const SimulationBufferView& source,
                StateBridge& bridge,
                opengl::Buffer& target) {
    if (source.bytes > 0 && source.data == nullptr) {
        throw std::logic_error("Render view contains a null required buffer.");
    }
    bridge.upload(source, target);
}

void
upload_optional(const std::optional<SimulationBufferView>& source,
                StateBridge& bridge,
                std::optional<opengl::Buffer>& target) {
    if (!source) {
        if (target) bridge.release(*target);
        target.reset();
        return;
    }
    if (source->bytes > 0 && source->data == nullptr) {
        throw std::logic_error("Render view contains a null optional buffer.");
    }
    if (!target) target.emplace();
    bridge.upload(*source, *target);
}

}

const RenderState&
RawStateProvider::update(const SimulationRenderView& view) {
    _state.particle_count = view.particle_count;
    upload_required(view.position, _bridge, _state.position);
    upload_required(view.velocity, _bridge, _state.velocity);
    upload_required(view.species, _bridge, _state.species);
    upload_optional(view.temperature, _bridge, _state.temperature);
    upload_optional(view.translational_energy, _bridge, _state.translational_energy);
    upload_optional(view.rotational_energy, _bridge, _state.rotational_energy);
    upload_optional(view.vibrational_energy, _bridge, _state.vibrational_energy);
    return _state;
}

}
