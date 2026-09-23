/**
 * @file
 * @brief Implements raw simulation-state uploads for rendering.
 */

#include "rendering/state/raw_state_provider.h"

#include "view/simulation_scene_view.h"

#include <optional>
#include <stdexcept>

namespace atlas::interactive {

namespace {

/// Validates and uploads one mandatory simulation state.
void
upload_required(const SimulationBufferView& source,
                StateBridge& bridge,
                opengl::Buffer& target) {
    if (source.bytes > 0 && source.data == nullptr) {
        throw std::logic_error("Render view contains a null required buffer.");
    }
    bridge.upload(source, target);
}

/// Mirrors the presence and contents of one optional simulation state.
void
upload_optional(const std::optional<SimulationBufferView>& source,
                StateBridge& bridge,
                std::optional<opengl::Buffer>& target) {
    if (!source) {
        // Drop both the OpenGL buffer and any backend registration when core state disappears.
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
RawStateProvider::update(const SimulationSceneView& view) {
    // Every upload is limited to the live prefix prepared by Session::render_view().
    _state.particle_count = view.particles.particle_count;
    upload_required(view.particles.position, _bridge, _state.position);
    upload_required(view.particles.velocity, _bridge, _state.velocity);
    upload_required(view.particles.species, _bridge, _state.species);
    upload_optional(view.particles.temperature, _bridge, _state.temperature);
    upload_optional(view.particles.translational_energy, _bridge, _state.translational_energy);
    upload_optional(view.particles.rotational_energy, _bridge, _state.rotational_energy);
    upload_optional(view.particles.vibrational_energy, _bridge, _state.vibrational_energy);
    _state.geometries = view.geometries;
    _state.lower_corner = view.lower_corner;
    _state.upper_corner = view.upper_corner;
    return _state;
}

}
