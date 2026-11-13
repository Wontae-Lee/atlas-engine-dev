#include "application/interactive_application.h"
#include "application/render_manager.h"
#include "transport/json_codec.h"

#include <charconv>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

std::size_t
maximum_steps(int argc, char** argv) {
    if (argc > 2) throw std::invalid_argument("usage: " + std::string(argv[0]) + " [steps]");
    if (argc == 1) return 0;
    const std::string_view value(argv[1]);
    std::size_t steps = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), steps);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size()) {
        throw std::invalid_argument("steps must be a nonnegative decimal integer (0 runs until window closes)");
    }
    return steps;
}

std::filesystem::path
configuration_path(const char* executable) {
    std::filesystem::path executable_path = std::filesystem::absolute(executable);
    if (std::filesystem::exists("/proc/self/exe")) {
        executable_path = std::filesystem::read_symlink("/proc/self/exe");
    }
    const auto directory                     = executable_path.parent_path();
    const std::filesystem::path candidates[] = {
        directory / "config.json",
        directory / ".." / "examples" / "interactive" / ATLAS_INTERACTIVE_EXAMPLE_NAME / "config.json"
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    throw std::runtime_error("Cannot find config.json for " ATLAS_INTERACTIVE_EXAMPLE_NAME
                             " beside the executable or in the installed examples directory");
}

std::string
read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open configuration: " + path.string());
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
        const auto steps = maximum_steps(argc, argv);
        auto simulation  = atlas::interactive::JsonCodec::decode_simulation(
            read_file(configuration_path(argv[0])));

        atlas::interactive::InteractiveApplication app(
            std::make_unique<atlas::interactive::RenderManager>());
        atlas::interactive::Request create;
        create.command     = atlas::interactive::Command::create;
        create.simulation  = std::move(simulation);
        const auto created = handle(app, create);
        if (!created.session_id) throw std::logic_error("create returned no session_id");

        atlas::interactive::Request command;
        command.session_id = *created.session_id;
        command.command    = atlas::interactive::Command::render_open;
        handle(app, command);
        command.command = atlas::interactive::Command::start;
        handle(app, command);

        while (app.rendering_active()) {
            app.update();
            if (steps > 0) {
                command.command   = atlas::interactive::Command::status;
                const auto status = handle(app, command);
                if (status.status && status.status->step >= steps) break;
            }
        }

        command.command = atlas::interactive::Command::pause;
        handle(app, command);
        command.command = atlas::interactive::Command::render_close;
        handle(app, command);
        command.command    = atlas::interactive::Command::step;
        command.step_count = 1;
        handle(app, command);
        command.command         = atlas::interactive::Command::status;
        const auto final_status = handle(app, command);
        if (final_status.status) {
            std::printf(ATLAS_INTERACTIVE_EXAMPLE_NAME ": completed steps=%zu particles=%zu\n",
                        final_status.status->step,
                        final_status.status->particle_count);
        }
        command.command = atlas::interactive::Command::close;
        handle(app, command);

        atlas::interactive::Request shutdown;
        shutdown.command = atlas::interactive::Command::shutdown;
        handle(app, shutdown);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, ATLAS_INTERACTIVE_EXAMPLE_NAME ": %s\n", error.what());
        return 1;
    }
}
