/**
 * @file
 * @brief Provides the native interactive executable entry point.
 */

#include "protocol/command.h"
#include "server/server.h"
#include "transport/json_codec.h"
#include "transport/json_lines_transport.h"

#include <atlas/logging/logging.h>

#include <cstdio>
#include <condition_variable>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
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
        atlas::interactive::Server server;
        atlas::interactive::JsonLinesTransport transport(std::cin, std::cout);

        if (argc == 3 && std::string(argv[1]) == "--config") {
            atlas::interactive::Request request;
            request.request_id = "startup";
            request.command = atlas::interactive::Command::create;
            request.simulation = atlas::interactive::JsonCodec::decode_simulation(
                read_file(argv[2]));
            transport.send(server.handle(request));
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

        while (!server.shutdown_requested()) {
            std::optional<atlas::interactive::Request> request;
            {
                std::unique_lock lock(request_mutex);
                // Sleep only when no simulation needs periodic advancement.
                if (requests.empty() && !input_closed && !server.has_running_sessions()) {
                    request_ready.wait(lock, [&] { return !requests.empty() || input_closed; });
                }
                if (!requests.empty()) {
                    request.emplace(std::move(requests.front()));
                    requests.pop_front();
                } else if (input_closed) {
                    break;
                }
            }
            if (request) transport.send(server.handle(*request));
            if (server.shutdown_requested()) break;
            server.update();
        }
        reader.join();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "atlas interactive: %s\n", error.what());
        return 1;
    }
}
