#include "rendering/layer/particle_layer.h"
#include "rendering/renderer.h"
#include "rendering/state/raw_state_provider.h"
#include "rendering/target/window_target.h"
#include "session/session.h"

#include <atlas/atlas.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <utility>

namespace {

atlas::System
make_system() {
    auto fluid = atlas::Fluid::builder()
                     .with_buffer_size(1)
                     .with_particle_count(1)
                     .make_host_unique();

    const atlas::Float3 position(-0.5f, 0.0f, 0.0f);
    const atlas::Float3 velocity(0.1f, 0.0f, 0.0f);
    atlas::copy_host_to_device(&position,
                               fluid->state<atlas::FluidPositionState>()->data(),
                               1);
    atlas::copy_host_to_device(&velocity,
                               fluid->state<atlas::FluidVelocityState>()->data(),
                               1);

    auto universe = atlas::Universe::builder()
                        .with_lower_corner(atlas::Float3(-1.0f))
                        .with_upper_corner(atlas::Float3(1.0f))
                        .with_cell_size(0.25f)
                        .make_host_unique();

    return atlas::System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_dt(0.01f)
        .build();
}

}

int
main(int argc, char** argv) {
    try {
        const std::size_t maximum_steps =
            argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 0;

        atlas::interactive::Session session(make_system());
        atlas::interactive::WindowTarget target(1280, 720, "Atlas");
        atlas::interactive::Renderer renderer(
            std::make_unique<atlas::interactive::RawStateProvider>());
        renderer.add_layer(std::make_unique<atlas::interactive::ParticleLayer>(8.0f));
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
        std::fprintf(stderr, "atlas interactive: %s\n", error.what());
        return 1;
    }
}
