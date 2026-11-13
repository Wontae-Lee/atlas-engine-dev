#include "application/interactive_application.h"
#include "application/render_manager.h"
#include "rendering/camera.h"
#include "transport/json_codec.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <utility>

int
main() {
    // Build outputs keep their JSON beside the executable. Installed binaries use
    // the matching examples directory. Resolve paths from the executable so relative
    // asset paths in JSON work regardless of the directory used to launch the program.
    const auto directory = std::filesystem::read_symlink("/proc/self/exe").parent_path();
    std::filesystem::current_path(
        std::filesystem::is_regular_file(directory / "config.json")
            ? directory
            : directory / ".." / "examples" / "interactive" / "00_template");

    // Keep the initial state, domain, timestep, and physical components in JSON.
    // Interactive constructs the Core system when the session is created below.
    std::ifstream input("config.json");
    std::ostringstream content;
    content << input.rdbuf();
    auto scene = atlas::interactive::JsonCodec::decode_simulation(content.str());

    // View the origin from +Z. The constructor takes the world-space position and
    // orbit target; its default up direction is +Y. Adjust these to frame a new scene.
    atlas::interactive::Camera camera(
        { 0.0f, 0.0f, 3.0f },
        { 0.0f, 0.0f, 0.0f });
    atlas::interactive::InteractiveApplication app(
        std::make_unique<atlas::interactive::RenderManager>(camera));

    // Create the simulation through Interactive. Its session owns the Core system;
    // rendering observes that system rather than advancing a separate simulation.
    atlas::interactive::Request create;
    create.command     = atlas::interactive::Command::create;
    create.simulation  = std::move(scene);
    const auto created = app.handle(create);
    if (!created.success || !created.session_id) return 1;

    // Route commands to this session. Open its window and draw the initial state
    // before starting time integration, so this first draw does not advance physics.
    atlas::interactive::Request command;
    command.session_id = *created.session_id;
    command.command    = atlas::interactive::Command::render_open;
    if (!app.handle(command).success) return 1;
    app.update();

    // Advance the session and render its current state until the window closes.
    // Each simulation update uses the JSON timestep; no CLI step limit or extra
    // simulation step is applied when the window closes.
    command.command = atlas::interactive::Command::start;
    if (!app.handle(command).success) return 1;
    while (app.rendering_active()) {
        app.update();
    }

    // The window and session have separate lifetimes. Close the remaining session
    // explicitly, then shut down the Interactive application.
    command.command = atlas::interactive::Command::close;
    if (!app.handle(command).success) return 1;
    atlas::interactive::Request shutdown;
    shutdown.command = atlas::interactive::Command::shutdown;
    return app.handle(shutdown).success ? 0 : 1;
}
