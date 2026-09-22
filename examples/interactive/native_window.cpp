#include "config/simulation_config.h"
#include "rendering/layer/geometry_layer.h"
#include "rendering/layer/particle_layer.h"
#include "rendering/renderer.h"
#include "rendering/state/raw_state_provider.h"
#include "rendering/target/window_target.h"
#include "session/session.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>

namespace {

atlas::interactive::SimulationConfig
make_config() {
    constexpr std::size_t particle_count = 64;
    atlas::interactive::SimulationConfig config;
    config.dt = 0.02f;
    config.fluid.buffer_size = particle_count;
    config.fluid.particle_count = particle_count;
    config.fluid.position.reserve(particle_count);
    config.fluid.velocity.reserve(particle_count);
    config.fluid.species.assign(particle_count, 0);
    atlas::interactive::SimulationConfig::Material material;
    material.kind = atlas::interactive::SimulationConfig::MaterialKind::molecule;
    material.mass = 4.65e-26f;
    material.reference_diameter = 3.7e-10f;
    material.reference_temperature = 273.0f;
    material.viscosity_index = 0.75f;
    config.fluid.materials.push_back(material);
    config.universe.lower_corner = { -2.0f, -2.0f, -2.0f };
    config.universe.upper_corner = { 2.0f, 2.0f, 2.0f };
    config.universe.cell_size = 0.5f;

    for (std::size_t index = 0; index < particle_count; ++index) {
        const float x = -0.7f + 0.2f * static_cast<float>(index % 8);
        const float y = -0.7f + 0.2f * static_cast<float>(index / 8);
        config.fluid.position.push_back({ x, y, 0.0f });
        config.fluid.velocity.push_back({ -0.15f * y, 0.15f * x, 0.0f });
    }
    atlas::interactive::SimulationConfig::Collider collider;
    collider.unit.geometry.kind = atlas::interactive::SimulationConfig::GeometryKind::sphere;
    collider.unit.geometry.radius = 0.35f;
    config.colliders.push_back(collider);
    config.solvers.emplace_back();
    return config;
}

}

int
main(int argc, char** argv) {
    try {
        const std::size_t maximum_steps =
            argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 0;

        atlas::interactive::Session session(make_config());
        atlas::interactive::WindowTarget target(1280, 720, "Atlas Interactive Example");
        atlas::interactive::Renderer renderer(
            std::make_unique<atlas::interactive::RawStateProvider>());
        renderer.add_layer(std::make_unique<atlas::interactive::GeometryLayer>());
        renderer.add_layer(std::make_unique<atlas::interactive::ParticleLayer>(6.0f));
        renderer.camera().set_position({ 0.0f, 0.0f, 3.0f });
        renderer.camera().set_target({ 0.0f, 0.0f, 0.0f });
        renderer.initialize(target);

        session.start();
        while (!target.should_close()
               && (maximum_steps == 0 || session.status().step < maximum_steps)) {
            target.poll_events(renderer.camera());
            session.update();
            renderer.render(session.scene_view(), target);
        }

        renderer.shutdown();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "interactive example: %s\n", error.what());
        return 1;
    }
}
