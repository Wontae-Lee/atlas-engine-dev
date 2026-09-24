/**
 * @file
 * @brief Exercises the Interactive application and its native rendering controls.
 */

#include "application/interactive_application.h"
#include "application/render_manager.h"
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

std::string
read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open interactive example case: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

atlas::interactive::Response
handle(atlas::interactive::InteractiveApplication& app,
       const atlas::interactive::Request& request) {
    auto response = app.handle(request);
    if (!response.success) throw std::runtime_error(response.error);
    return response;
}

}

int
main(int argc, char** argv) {
    try {
        const std::string selection = argc > 1 ? argv[1] : "cylinder";
        const std::size_t maximum_steps =
            argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 0;
        auto simulation = atlas::interactive::JsonCodec::decode_simulation(
            read_file(case_path(selection, argv[0])));

        atlas::interactive::InteractiveApplication app(
            std::make_unique<atlas::interactive::RenderManager>());
        atlas::interactive::Request create;
        create.command = atlas::interactive::Command::create;
        create.simulation = std::move(simulation);
        const auto created = handle(app, create);
        if (!created.session_id) throw std::logic_error("create returned no session_id");

        atlas::interactive::Request command;
        command.session_id = *created.session_id;
        command.command = atlas::interactive::Command::render_open;
        handle(app, command);
        command.command = atlas::interactive::Command::start;
        handle(app, command);

        while (app.rendering_active()) {
            app.update();
            if (maximum_steps > 0) {
                command.command = atlas::interactive::Command::status;
                const auto status = handle(app, command);
                if (status.status && status.status->step >= maximum_steps) break;
            }
        }

        command.command = atlas::interactive::Command::pause;
        handle(app, command);
        command.command = atlas::interactive::Command::render_close;
        handle(app, command);
        command.command = atlas::interactive::Command::step;
        command.step_count = 1;
        handle(app, command);
        command.command = atlas::interactive::Command::close;
        handle(app, command);

        atlas::interactive::Request shutdown;
        shutdown.command = atlas::interactive::Command::shutdown;
        handle(app, shutdown);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "interactive example: %s\n", error.what());
        return 1;
    }
}
