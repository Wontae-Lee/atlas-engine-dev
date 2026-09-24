/**
 * @file
 * @brief Provides the native interactive executable entry point.
 */

#include "application/interactive_application.h"
#if defined(ATLAS_INTERACTIVE_RENDERING_ENABLED)
#include "application/render_manager.h"
#endif
#include "protocol/command.h"
#include "transport/json_codec.h"
#include "transport/json_lines_transport.h"

#include <atlas/logging/logging.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

/// Reads an entire simulation configuration file as UTF-8 text.
std::string
read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open simulation config: " + path);
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

}

int
main(int argc, char** argv) {
    try {
        atlas::Logging::set_all_stream(&std::cerr);
#if defined(ATLAS_INTERACTIVE_RENDERING_ENABLED)
        atlas::interactive::InteractiveApplication app(
            std::make_unique<atlas::interactive::RenderManager>());
#else
        atlas::interactive::InteractiveApplication app;
#endif
        atlas::interactive::JsonLinesTransport transport(std::cin, std::cout);

        if (argc == 3 && std::string(argv[1]) == "--config") {
            atlas::interactive::Request request;
            request.request_id = "startup";
            request.command = atlas::interactive::Command::create;
            request.simulation = atlas::interactive::JsonCodec::decode_simulation(
                read_file(argv[2]));
            transport.send(app.handle(request));
        } else if (argc != 1) {
            throw std::invalid_argument("Usage: atlas-interactive [--config simulation.json]");
        }

        std::deque<atlas::interactive::Request> requests;
        std::mutex request_mutex;
        std::condition_variable request_ready;
        bool input_closed = false;

        // Blocking input runs separately so running sessions can continue advancing.
        std::thread reader([&] {
            atlas::interactive::Request request;
            atlas::interactive::Response parse_error;
            while (transport.receive(request, parse_error)) {
                const bool shutdown = request.command == atlas::interactive::Command::shutdown;
                {
                    const std::lock_guard lock(request_mutex);
                    requests.push_back(std::move(request));
                }
                request_ready.notify_one();
                if (shutdown) break;
            }
            {
                const std::lock_guard lock(request_mutex);
                input_closed = true;
            }
            request_ready.notify_one();
        });

        while (!app.shutdown_requested()) {
            std::optional<atlas::interactive::Request> request;
            {
                std::unique_lock lock(request_mutex);
                if (requests.empty() && !input_closed) {
                    if (!app.has_running_sessions() && !app.rendering_active()) {
                        request_ready.wait(lock, [&] { return !requests.empty() || input_closed; });
                    } else if (!app.has_running_sessions()) {
                        request_ready.wait_for(lock, std::chrono::milliseconds(16), [&] {
                            return !requests.empty() || input_closed;
                        });
                    }
                }
                if (!requests.empty()) {
                    request.emplace(std::move(requests.front()));
                    requests.pop_front();
                } else if (input_closed) {
                    break;
                }
            }
            if (request) transport.send(app.handle(*request));
            if (app.shutdown_requested()) break;
            app.update();
        }
        reader.join();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "atlas interactive: %s\n", error.what());
        return 1;
    }
}
