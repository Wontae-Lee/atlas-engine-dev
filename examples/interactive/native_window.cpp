#include "rendering/layer/particle_layer.h"
#include "rendering/renderer.h"
#include "rendering/state/raw_state_provider.h"
#include "rendering/target/window_target.h"
#include "session/session.h"

#include <atlas/atlas.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <utility>

namespace {

atlas::System
make_system() {
    constexpr std::size_t particle_count = 64;
    std::array<atlas::Float3, particle_count> positions;
    std::array<atlas::Float3, particle_count> velocities;

    for (std::size_t index = 0; index < particle_count; ++index) {
        const float x = -0.7f + 0.2f * static_cast<float>(index % 8);
        const float y = -0.7f + 0.2f * static_cast<float>(index / 8);
        positions[index] = atlas::Float3(x, y, 0.0f);
        velocities[index] = atlas::Float3(-0.15f * y, 0.15f * x, 0.0f);
    }

    auto fluid = atlas::Fluid::builder()
                     .with_buffer_size(particle_count)
                     .with_particle_count(particle_count)
                     .make_host_unique();
    atlas::copy_host_to_device(positions.data(),
                               fluid->state<atlas::FluidPositionState>()->data(),
                               particle_count);
    atlas::copy_host_to_device(velocities.data(),
                               fluid->state<atlas::FluidVelocityState>()->data(),
                               particle_count);

    auto universe = atlas::Universe::builder()
                        .with_lower_corner(atlas::Float3(-2.0f))
                        .with_upper_corner(atlas::Float3(2.0f))
                        .with_cell_size(0.5f)
                        .make_host_unique();

    return atlas::System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_dt(0.02f)
        .build();
}

}

int
main(int argc, char** argv) {
    try {
        const std::size_t maximum_steps =
            argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 0;

        atlas::interactive::Session session(make_system());
        atlas::interactive::WindowTarget target(1280, 720, "Atlas Interactive Example");
        atlas::interactive::Renderer renderer(
            std::make_unique<atlas::interactive::RawStateProvider>());
        renderer.add_layer(std::make_unique<atlas::interactive::ParticleLayer>(6.0f));
        renderer.camera().set_position({ 0.0f, 0.0f, 3.0f });
        renderer.camera().set_target({ 0.0f, 0.0f, 0.0f });
        renderer.initialize(target);

        session.start();
        while (!target.should_close()
               && (maximum_steps == 0 || session.system().step() < maximum_steps)) {
            target.poll_events();
            session.update();
            renderer.render(session.system(), target);
        }

        renderer.shutdown();
        session.close();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "interactive example: %s\n", error.what());
        return 1;
    }
}
