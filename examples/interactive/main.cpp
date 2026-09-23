/**
 * @file
 * @brief Runs a configured Atlas simulation in the native interactive renderer.
 */

#include "rendering/layer/geometry_layer.h"
#include "rendering/layer/particle_layer.h"
#include "rendering/renderer.h"
#include "rendering/state/raw_state_provider.h"
#include "rendering/target/window_target.h"
#include "session/session.h"
#include "transport/json_codec.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

/**
 * @brief Resolves a case name or explicit path to a simulation JSON file.
 * @param selection Case name or filesystem path supplied by the user.
 * @param executable Process executable path used to locate installed cases.
 * @return Existing configuration path.
 * @throws std::invalid_argument when no matching case exists.
 */
std::filesystem::path
case_path(const std::string& selection, const char* executable) {
    const std::filesystem::path explicit_path(selection);
    if (std::filesystem::exists(explicit_path)) return explicit_path;

    std::filesystem::path executable_path = std::filesystem::absolute(executable);
    if (std::filesystem::exists("/proc/self/exe")) {
        executable_path = std::filesystem::read_symlink("/proc/self/exe");
    }
    const std::filesystem::path executable_directory = executable_path.parent_path();
    const std::filesystem::path filename = selection + ".json";
    // Search beside the executable first, then source and configured install locations.
    const std::filesystem::path candidates[] = {
        executable_directory / "cases" / filename,
        executable_directory / ".." / "examples" / "interactive" / "cases" / filename,
        std::filesystem::path(ATLAS_INTERACTIVE_EXAMPLE_CASES_DIR) / filename
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) return candidate;
    }
    throw std::invalid_argument("Unknown interactive example case: " + selection);
}

/// Reads an entire interactive case file for JsonCodec.
std::string
read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open interactive example case: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

}

/// Builds the raw every-step rendering pipeline and owns its native event loop.
int
main(int argc, char** argv) {
    try {
        const std::string selection = argc > 1 ? argv[1] : "cylinder";
        const std::size_t maximum_steps =
            argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 0;
        const auto simulation = atlas::interactive::JsonCodec::decode_simulation(
            read_file(case_path(selection, argv[0])));

        atlas::interactive::Session session(simulation);
        atlas::interactive::WindowTarget target(1280, 720, "Atlas Interactive Example");
        atlas::interactive::Renderer renderer(
            std::make_unique<atlas::interactive::RawStateProvider>());
        renderer.add_layer(std::make_unique<atlas::interactive::GeometryLayer>());
        renderer.add_layer(std::make_unique<atlas::interactive::ParticleLayer>(6.0f));
        renderer.camera().set_position({ 0.0f, 0.0f, 3.0f });
        renderer.camera().set_target({ 0.0f, 0.0f, 0.0f });
        renderer.initialize(target);

        session.start();
        // Raw mode intentionally advances exactly once before every rendered frame.
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
