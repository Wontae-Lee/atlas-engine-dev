#include "application/interactive_application.h"
#include "application/render_manager.h"
#include "rendering/camera.h"
#include "transport/json_codec.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>

int
main() {
    // Display one complete wing cycle over roughly five wall-clock seconds.
    // This controls viewing speed only; the JSON timestep and motion stay unchanged.
    constexpr float display_seconds_per_cycle = 5.0f;

    // Build outputs keep their JSON beside the executable. Installed binaries use
    // ../examples/interactive/02_intake instead. Resolve the directory from the
    // executable so the JSON's relative OBJ paths work regardless of launch location.
    const auto directory = std::filesystem::read_symlink("/proc/self/exe").parent_path();
    std::filesystem::current_path(
        std::filesystem::is_regular_file(directory / "config_2.json")
            ? directory
            : directory / ".." / "examples" / "interactive" / "02_intake");

    // Case 2 produces a traveling wave; change to config_1.json for synchronized wings.
    // The JSON supplies mesh paths, hinge translations, motion, and flow settings.
    // Mesh loading and Core construction happen when Interactive creates the session.
    std::ifstream input("config_2.json");
    std::ostringstream content;
    content << input.rdbuf();
    auto scene = atlas::interactive::JsonCodec::decode_simulation(content.str());

    // Collider 0 is the fixed support; collider 1 is the first wing.
    // All wings share a period. Each update advances dt / period of one cycle,
    // so scale that fraction by the desired wall-clock duration to pace the display.
    const auto period   = scene.colliders.at(1).unit.angular_oscillation.value().period;
    const auto interval = std::chrono::duration<float>(display_seconds_per_cycle * scene.dt / period);

    // Frame the intake from an elevated oblique position, looking toward its center.
    // Camera arguments are world-space position followed by the orbit target;
    // the constructor's default up direction is +Y.
    atlas::interactive::Camera camera({ 3.4f, 3.2f, 4.3f }, { 0.0f, 0.75f, 0.0f });
    atlas::interactive::InteractiveApplication app(std::make_unique<atlas::interactive::RenderManager>(camera));

    // Create the simulation through Interactive. Its session owns the Core system;
    // the renderer observes that system rather than advancing a separate simulation.
    atlas::interactive::Request create;
    create.command     = atlas::interactive::Command::create;
    create.simulation  = std::move(scene);
    const auto created = app.handle(create);
    if (!created.success || !created.session_id) return 1;

    // Address subsequent commands to the created session. Open its window and draw
    // the configured initial pose before starting time integration.
    atlas::interactive::Request command;
    command.session_id = *created.session_id;
    command.command    = atlas::interactive::Command::render_open;
    if (!app.handle(command).success) return 1;
    app.update();

    // Run until the window closes. Each update advances the session and then renders
    // its current state. Sleep only for the time remaining in this display interval;
    // if computation takes longer, playback slows without changing the timestep.
    command.command = atlas::interactive::Command::start;
    if (!app.handle(command).success) return 1;
    while (app.rendering_active()) {
        const auto next_update = std::chrono::steady_clock::now() + interval;
        app.update();
        std::this_thread::sleep_until(next_update);
    }

    // Closing the window releases rendering resources but leaves the session alive.
    // Close the session explicitly, then shut down the Interactive application.
    command.command = atlas::interactive::Command::close;
    if (!app.handle(command).success) return 1;
    atlas::interactive::Request shutdown;
    shutdown.command = atlas::interactive::Command::shutdown;
    return app.handle(shutdown).success ? 0 : 1;
}
