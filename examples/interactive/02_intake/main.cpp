#include "application/interactive_application.h"
#include "application/render_manager.h"
#include "rendering/camera.h"
#include "transport/json_codec.h"

#include <atlas/math/constants.h>

#include <nlohmann/json.hpp>
#include <tiny_obj_loader.h>

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using Json = nlohmann::json;
using Vec3 = std::array<float, 3>;

struct Options {
    std::size_t steps = 0;
    bool headless = false;
    bool paced = true;
};

void
usage(const char* executable) {
    std::printf("Usage: %s [steps] [--headless] [--unpaced]\n"
                "Default: repeat until the window closes; 0 means unlimited.\n"
                "--headless requires a positive step count.\n", executable);
}

Options
options(int argc, char** argv) {
    Options result;
    bool supplied_steps = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view value(argv[index]);
        if (value == "--headless") result.headless = true;
        else if (value == "--unpaced") result.paced = false;
        else {
            if (supplied_steps) throw std::invalid_argument("expected only one step count");
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result.steps);
            if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
                throw std::invalid_argument("expected a nonnegative step count, --headless or --unpaced");
            }
            supplied_steps = true;
        }
    }
    if (result.headless && result.steps == 0) {
        throw std::invalid_argument("--headless requires a positive step count");
    }
    return result;
}

std::filesystem::path
example_directory(const char* executable) {
    auto path = std::filesystem::absolute(executable);
    if (std::filesystem::exists("/proc/self/exe")) path = std::filesystem::read_symlink("/proc/self/exe");
    const auto directory = path.parent_path();
    for (const auto& candidate : { directory, directory / ".." / "examples" / "interactive" / "02_intake" }) {
        if (std::filesystem::is_regular_file(candidate / "config.json")) return candidate;
    }
    throw std::runtime_error("Cannot find 02_intake/config.json beside the executable or in the installed examples");
}

Json
mesh_geometry(const std::filesystem::path& path, const Vec3& pivot) {
    tinyobj::ObjReaderConfig settings;
    settings.triangulate = true;
    settings.mtl_search_path = path.parent_path().string();
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path.string(), settings)) {
        throw std::runtime_error("Cannot read " + path.string() + ": " + reader.Error());
    }
    const auto& vertices = reader.GetAttrib().vertices;
    Json triangles = Json::array();
    for (const auto& shape : reader.GetShapes()) {
        std::size_t offset = 0;
        for (const auto count : shape.mesh.num_face_vertices) {
            if (count != 3) throw std::runtime_error("Expected triangulated OBJ faces in " + path.string());
            Json triangle = Json::array();
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const int vertex = shape.mesh.indices.at(offset + corner).vertex_index;
                if (vertex < 0 || static_cast<std::size_t>(vertex) >= vertices.size() / 3) {
                    throw std::runtime_error("Invalid OBJ vertex index in " + path.string());
                }
                const auto start = static_cast<std::size_t>(vertex) * 3;
                triangle.push_back(Vec3 { static_cast<float>(vertices[start]) - pivot[0],
                                          static_cast<float>(vertices[start + 1]) - pivot[1],
                                          static_cast<float>(vertices[start + 2]) - pivot[2] });
            }
            triangles.push_back(std::move(triangle));
            offset += 3;
        }
    }
    if (triangles.empty()) throw std::runtime_error("OBJ has no triangles: " + path.string());
    return { { "type", "triangle_mesh" }, { "triangles", std::move(triangles) } };
}

atlas::interactive::SimulationConfig
simulation(const Json& config, const std::filesystem::path& directory) {
    Json scene = config.at("simulation");
    const auto& motion = config.at("motion");
    const double amplitude = motion.at("closing_angle_degrees").get<double>() * atlas::pi_double / 180.0;
    const double period = motion.at("period_seconds").get<double>();
    const Json smooth = motion.value("smooth", Json(true));
    auto& colliders = scene["colliders"] = Json::array();
    colliders.push_back({ { "unit", { { "geometry", mesh_geometry(directory / "assets" / "support.obj", { 0, 0, 0 }) } } } });
    const float hinge_radius = motion.at("hinge_radius").get<float>();
    const float hinge_height = motion.at("hinge_height").get<float>();
    if (!std::isfinite(hinge_radius) || hinge_radius <= 0 || !std::isfinite(hinge_height)) {
        throw std::invalid_argument("hinge_radius must be finite and positive; hinge_height must be finite");
    }
    for (int wing = 1; wing <= 12; ++wing) {
        const double azimuth = static_cast<double>(wing - 1) * atlas::pi_double / 6.0;
        const float sine = static_cast<float>(std::sin(azimuth));
        const float cosine = static_cast<float>(std::cos(azimuth));
        const Vec3 pivot { hinge_radius * sine, hinge_height, hinge_radius * cosine };
        Json unit {
            { "geometry", mesh_geometry(directory / "assets" / ("wing" + std::to_string(wing) + ".obj"), pivot) },
            { "translation", pivot },
            { "angular_oscillation", {
                { "axis", Vec3 { -cosine, 0.0f, sine } },
                { "amplitude", amplitude },
                { "period", period },
                { "smooth", smooth }
            } }
        };
        colliders.push_back({ { "unit", std::move(unit) },
                              { "momentum_accommodation_coefficient", 1.0 },
                              { "restitution", 1.0 } });
    }
    return atlas::interactive::JsonCodec::decode_simulation(scene.dump());
}

atlas::interactive::Response
handle(atlas::interactive::InteractiveApplication& app, const atlas::interactive::Request& request) {
    auto response = app.handle(request);
    if (!response.success) throw std::runtime_error(response.error);
    return response;
}

}

int
main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        usage(argv[0]);
        return 0;
    }
    try {
        const auto run = options(argc, argv);
        const auto directory = example_directory(argv[0]);
        std::ifstream input(directory / "config.json");
        if (!input) throw std::runtime_error("Cannot read intake config.json");
        const Json config = Json::parse(input);
        const double display_period = config.at("motion").at("display_seconds_per_cycle").get<double>();
        if (!(display_period > 0.0) || !std::isfinite(display_period)) {
            throw std::invalid_argument("display_seconds_per_cycle must be finite and positive");
        }
        auto scene = simulation(config, directory);
        const double dt = scene.dt;
        const double period = config.at("motion").at("period_seconds").get<double>();
        const double amplitude = config.at("motion").at("closing_angle_degrees").get<double>();
        const bool smooth = config.at("motion").value("smooth", true);

        std::unique_ptr<atlas::interactive::RenderControl> rendering;
        if (!run.headless) {
            atlas::interactive::Camera camera;
            camera.set_target({ 0.0f, 0.75f, 0.0f });
            camera.set_position({ 3.4f, 3.2f, 4.3f });
            rendering = std::make_unique<atlas::interactive::RenderManager>(camera);
        }
        atlas::interactive::InteractiveApplication app(std::move(rendering));
        atlas::interactive::Request create;
        create.command = atlas::interactive::Command::create;
        create.simulation = std::move(scene);
        const auto created = handle(app, create);
        if (!created.session_id) throw std::logic_error("create returned no session_id");
        const double interval_seconds = display_period * dt / period;
        if (!std::isfinite(interval_seconds) || interval_seconds > 60.0) {
            throw std::invalid_argument("display pacing exceeds 60 seconds per update; reduce display_seconds_per_cycle");
        }
        const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(interval_seconds));
        atlas::interactive::Request command;
        command.session_id = *created.session_id;
        if (!run.headless) {
            command.command = atlas::interactive::Command::render_open;
            handle(app, command);
            app.update();
        }
        command.command = atlas::interactive::Command::start;
        handle(app, command);
        std::printf("02_intake: 12 wings + support, close=%.1fdeg period=%.6fs smooth=%s dt=%.1es\n",
                    amplitude, period, smooth ? "true" : "false", dt);
        double next_report_time = period * 0.5;
        std::size_t completed = 0;
        while ((run.headless || app.rendering_active()) && (run.steps == 0 || completed < run.steps)) {
            const auto next_update = std::chrono::steady_clock::now() + interval;
            app.update();
            ++completed;
            if (static_cast<double>(completed) * dt >= next_report_time - dt * 0.01 || completed == run.steps) {
                next_report_time += period * 0.5;
                command.command = atlas::interactive::Command::status;
                const auto status = handle(app, command);
                if (status.status) {
                    const double phase = std::fmod(status.status->simulation_time, period) / period;
                    const double angle = smooth
                                             ? amplitude * 0.5 * (1.0 - std::cos(2.0 * atlas::pi_double * phase))
                                             : 2.0 * amplitude * (phase < 0.5 ? phase : 1.0 - phase);
                    std::printf("step=%zu angle=%.2fdeg particles=%zu\n", status.status->step,
                                angle, status.status->particle_count);
                }
            }
            if (!run.headless && run.paced) std::this_thread::sleep_until(next_update);
        }
        command.command = atlas::interactive::Command::pause;
        handle(app, command);
        if (!run.headless) {
            command.command = atlas::interactive::Command::render_close;
            handle(app, command);
        }
        command.command = atlas::interactive::Command::status;
        const auto final = handle(app, command);
        if (final.status) std::printf("02_intake: completed steps=%zu particles=%zu\n", final.status->step, final.status->particle_count);
        command.command = atlas::interactive::Command::close;
        handle(app, command);
        atlas::interactive::Request shutdown;
        shutdown.command = atlas::interactive::Command::shutdown;
        handle(app, shutdown);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "02_intake: %s\n", error.what());
        return 1;
    }
}
