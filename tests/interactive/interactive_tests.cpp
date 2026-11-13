#include "config/output_config.h"
#include "config/simulation_config.h"
#include "config/system_factory.h"
#include "protocol/command.h"
#include "protocol/request.h"
#include "server/server.h"
#include "session/session.h"
#include "session/session_state.h"
#include "transport/json_codec.h"
#include "transport/json_lines_transport.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>

namespace {

using Config = atlas::interactive::SimulationConfig;

Config
make_config() {
    Config config;
    config.dt = 0.25f;
    config.fluid.buffer_size = 4;
    config.fluid.particle_count = 1;
    config.fluid.position = { Config::Vec3 { 0.0f, 0.0f, 0.0f } };
    config.fluid.velocity = { Config::Vec3 { 1.0f, 0.0f, 0.0f } };
    config.fluid.species = { 0 };
    config.universe.lower_corner = { -2.0f, -2.0f, -2.0f };
    config.universe.upper_corner = { 2.0f, 2.0f, 2.0f };
    config.universe.cell_size = 1.0f;
    return config;
}

Config
make_observed_config() {
    Config config = make_config();
    config.dt = 0.01f;
    config.fluid.buffer_size = 64;
    config.fluid.particle_count = 0;
    config.fluid.position.clear();
    config.fluid.velocity.clear();
    config.fluid.species.clear();

    Config::Emitter emitter;
    emitter.source.kind = Config::SourceKind::volume;
    emitter.source.spacing = 0.5f;
    emitter.source.unit.geometry.kind = Config::GeometryKind::box;
    emitter.source.unit.geometry.lower_corner = { -0.5f, -0.5f, -0.5f };
    emitter.source.unit.geometry.upper_corner = { 0.5f, 0.5f, 0.5f };
    emitter.generator.kind = Config::GeneratorKind::uniform;
    emitter.generator.min_value = 0.0f;
    emitter.generator.max_value = 0.0f;
    config.emitters.push_back(emitter);

    Config::Sink sink;
    sink.kind = Config::SinkKind::volume;
    sink.unit.geometry.kind = Config::GeometryKind::box;
    sink.unit.geometry.lower_corner = { -1.0f, -1.0f, -1.0f };
    sink.unit.geometry.upper_corner = { 1.0f, 1.0f, 1.0f };
    config.sinks.push_back(sink);
    return config;
}

Config
decode_oscillating_config(const std::string& motion) {
    return atlas::interactive::JsonCodec::decode_simulation(
        R"({"dt":0.25,"fluid":{"buffer_size":4},
            "universe":{"lower_corner":[-3,-3,-3],"upper_corner":[3,3,3],"cell_size":1},
            "colliders":[{"unit":{
                "geometry":{"type":"box","lower_corner":[0,0,0],"upper_corner":[1,0.1,0.1]},
                "translation":[1,0.5,0],"angular_oscillation":)"
        + motion + "}}]}");
}

atlas::interactive::Request
request(const atlas::interactive::Command command, const std::uint64_t session_id) {
    atlas::interactive::Request result;
    result.command = command;
    result.session_id = session_id;
    return result;
}

}

TEST(InteractiveConfig, JsonParsesSimulationAndRejectsMalformedInput) {
    const std::string json = R"({
        "dt": 0.25,
        "fluid": {
            "buffer_size": 4,
            "particle_count": 1,
            "position": [[0, 0, 0]],
            "velocity": [[1, 0, 0]],
            "species": [0],
            "temperature": [300]
        },
        "universe": {
            "lower_corner": [-2, -2, -2],
            "upper_corner": [2, 2, 2],
            "cell_size": 1
        },
        "solvers": [{"kernel": "variable_soft_sphere"}],
        "colliders": [{
            "unit": {"geometry": {"type": "sphere", "radius": 0.5}},
            "diffuse_sampling": "cosine_weighted"
        }],
        "codec": {
            "representative_characteristic_length": 1,
            "representative_collision_cross_sectional_area": 1,
            "representative_statistical_weight": 1,
            "representative_cell_volume": 1
        }
    })";
    const Config config = atlas::interactive::JsonCodec::decode_simulation(json);
    EXPECT_FLOAT_EQ(config.dt, 0.25f);
    EXPECT_EQ(config.fluid.position.size(), 1u);
    ASSERT_EQ(config.solvers.size(), 1u);
    EXPECT_EQ(config.solvers[0].kernel, Config::DsmcKernelKind::variable_soft_sphere);
    ASSERT_EQ(config.colliders.size(), 1u);
    EXPECT_EQ(config.colliders[0].unit.geometry.kind, Config::GeometryKind::sphere);
    EXPECT_TRUE(config.codec.has_value());
    EXPECT_THROW(atlas::interactive::JsonCodec::decode_simulation("{"), std::exception);
}

TEST(InteractiveConfig, JsonCoversAllUserFacingConstructionBranches) {
    const Config config = atlas::interactive::JsonCodec::decode_simulation(R"({
        "dt": 0.01,
        "fluid": {
            "buffer_size": 8,
            "particle_count": 1,
            "statistical_weight": 2,
            "materials": [
                {"type":"molecule","mass":1,"reference_diameter":1,"reference_temperature":1},
                {"type":"atom","mass":1,"reference_diameter":1,"reference_temperature":1},
                {"type":"ion","mass":1,"reference_diameter":1,"reference_temperature":1},
                {"type":"neutron","mass":1,"reference_diameter":1,"reference_temperature":1},
                {"type":"solid","mass":1}
            ],
            "position": [[0,0,0]],
            "velocity": [[1,2,3]],
            "species": [0],
            "temperature": [300],
            "translational_energy": [1],
            "rotational_energy": [2],
            "vibrational_energy": [3]
        },
        "universe": {
            "geometry": {"type":"box","lower_corner":[-1,-1,-1],"upper_corner":[1,1,1]},
            "cell_size": 1,
            "temperature": [300,300,300,300,300,300,300,300],
            "bulk_velocity": [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
            "field_force": [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
            "gravity": [[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1]],
            "thermal_energy": [1,1,1,1,1,1,1,1],
            "knudsen_number": [1,1,1,1,1,1,1,1]
        },
        "solvers": [
            {"kernel":"hard_sphere"},
            {"kernel":"variable_hard_sphere"},
            {"kernel":"variable_soft_sphere"}
        ],
        "emitters": [
            {"source":{"type":"surface","unit":{"geometry":{"type":"circle"}}},"generator":{"type":"uniform"}},
            {"source":{"type":"volume","unit":{"geometry":{"type":"sphere"}}},"generator":{"type":"jittering"}},
            {"source":{"type":"volume","unit":{"geometry":{"type":"cylinder"}}},"generator":{"type":"maxwell_sigma"}},
            {"source":{"type":"volume","unit":{"geometry":{"type":"box"}}},"generator":{"type":"maxwell_boltzmann","species_mass":[1]}}
        ],
        "colliders": [{
            "unit": {
                "geometry":{"type":"square"},
                "translation":[1,2,3],
                "orientation":[1,0,0,0],
                "velocity":[1,0,0],
                "acceleration":[0,1,0],
                "angular_velocity":[0,0,1],
                "angular_acceleration":[0,0,2]
            },
            "momentum_accommodation_coefficient":0.5,
            "restitution":0.8,
            "diffuse_sampling":"cosine_weighted"
        }],
        "sinks": [
            {"type":"surface","unit":{"geometry":{"type":"plane"}}},
            {"type":"volume","unit":{"geometry":{"type":"polygonal_prism"}}},
            {"type":"tracing","unit":{"geometry":{"type":"triangle_mesh","triangles":[[[0,0,0],[1,0,0],[0,1,0]]]}}}
        ],
        "codec": {
            "representative_characteristic_length":1,
            "representative_collision_cross_sectional_area":1,
            "representative_statistical_weight":1,
            "representative_cell_volume":1
        }
    })");

    EXPECT_EQ(config.fluid.materials.size(), 5u);
    EXPECT_TRUE(config.fluid.vibrational_energy.has_value());
    EXPECT_TRUE(config.universe.geometry.has_value());
    EXPECT_TRUE(config.universe.gravity.has_value());
    EXPECT_EQ(config.solvers.size(), 3u);
    ASSERT_EQ(config.emitters.size(), 4u);
    EXPECT_EQ(config.emitters[0].source.kind, Config::SourceKind::surface);
    EXPECT_EQ(config.emitters[3].generator.kind, Config::GeneratorKind::maxwell_boltzmann);
    EXPECT_EQ(config.emitters[3].generator.species_mass.size(), 1u);
    ASSERT_EQ(config.colliders.size(), 1u);
    EXPECT_TRUE(config.colliders[0].unit.angular_acceleration.has_value());
    ASSERT_EQ(config.sinks.size(), 3u);
    EXPECT_EQ(config.sinks[2].kind, Config::SinkKind::tracing);
    EXPECT_TRUE(config.codec.has_value());
}

TEST(InteractiveConfig, SystemFactoryBuildsAndRebuildsAValidSystem) {
    atlas::interactive::SystemFactory factory(make_config());
    auto first = factory.create();
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->fluid()->particle_count(), 1u);
    first->update();
    EXPECT_EQ(first->step(), 1u);

    auto second = factory.create();
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->step(), 0u);
    EXPECT_EQ(second->fluid()->particle_count(), 1u);
}

TEST(InteractiveConfig, AngularOscillationRequiresItsWireFields) {
    const auto config = decode_oscillating_config(
        R"({"axis":[0,0,1],"amplitude":1.57079632679,"period":2})");
    ASSERT_EQ(config.colliders.size(), 1u);
    ASSERT_TRUE(config.colliders[0].unit.angular_oscillation.has_value());
    const auto& motion = *config.colliders[0].unit.angular_oscillation;
    EXPECT_FLOAT_EQ(motion.axis[2], 1.0f);
    EXPECT_NEAR(motion.amplitude, 1.57079632679f, 1e-6f);
    EXPECT_FLOAT_EQ(motion.period, 2.0f);
    EXPECT_TRUE(motion.smooth);
    EXPECT_FLOAT_EQ(motion.elapsed, 0.0f);

    for (const std::string malformed : {
             R"({"amplitude":1,"period":2})",
             R"({"axis":[0,0,1],"period":2})",
             R"({"axis":[0,0,1],"amplitude":1})",
             R"({"axis":[0,1],"amplitude":1,"period":2})",
             R"({"axis":[0,0,1],"amplitude":"wide","period":2})",
             R"({"axis":[0,0,1],"amplitude":1,"period":null})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"smooth":0})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"smooth":"false"})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"smooth":null})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"elapsed":"0.25"})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"elapsed":true})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"elapsed":null})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"elapsed":[]})",
             "[]", "null" }) {
        SCOPED_TRACE(malformed);
        EXPECT_THROW(decode_oscillating_config(malformed), std::exception);
    }
}

TEST(InteractiveConfig, AngularOscillationSmoothOptionReachesCore) {
    for (const std::string option : { "", ",\"smooth\":true", ",\"smooth\":false" }) {
        SCOPED_TRACE(option);
        const Config config = decode_oscillating_config(
            R"({"axis":[0,0,1],"amplitude":1,"period":2)" + option + "}");
        const bool smooth = option != ",\"smooth\":false";
        const auto& configured_unit = config.colliders[0].unit;
        ASSERT_TRUE(configured_unit.angular_oscillation.has_value());
        EXPECT_EQ(configured_unit.angular_oscillation->smooth, smooth);

        atlas::interactive::CoreFactory factory;
        auto unit = factory.build(configured_unit);
        ASSERT_TRUE(unit.angular_velocity().has_value());
        EXPECT_FLOAT_EQ(unit.angular_velocity()->z, smooth ? 0.0f : 1.0f);
        unit.update(0.25f);
        const float expected_angle = smooth ? 0.1464466094f : 0.25f;
        const auto direction = unit.sync().sync_dir_to_world(atlas::Float3(1.0f, 0.0f, 0.0f));
        EXPECT_NEAR(direction.x, std::cos(expected_angle), 1e-6f);
        EXPECT_NEAR(direction.y, std::sin(expected_angle), 1e-6f);
    }
}

TEST(InteractiveConfig, AngularOscillationSemanticValidationComesFromCore) {
    for (const std::string invalid : {
             R"({"axis":[0,0,0],"amplitude":1,"period":2})",
             R"({"axis":[0,0,1],"amplitude":1,"period":0})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"elapsed":-0.25})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"elapsed":2})",
             R"({"axis":[0,0,1],"amplitude":1,"period":2,"elapsed":3})" }) {
        SCOPED_TRACE(invalid);
        const Config config = decode_oscillating_config(invalid);
        atlas::interactive::CoreFactory factory(config);
        EXPECT_THROW(factory.create(), std::exception);
    }

    auto conflict = decode_oscillating_config(
        R"({"axis":[0,0,1],"amplitude":1,"period":2})");
    conflict.colliders[0].unit.angular_velocity = Config::Vec3 { 0.0f, 0.0f, 1.0f };
    atlas::interactive::CoreFactory factory(conflict);
    EXPECT_THROW(factory.create(), std::exception);
}

TEST(InteractiveConfig, ExplicitEmptyOptionalFluidStateRemainsAvailable) {
    Config config = make_config();
    config.fluid.particle_count = 0;
    config.fluid.position.clear();
    config.fluid.velocity.clear();
    config.fluid.species.clear();
    config.fluid.temperature = std::vector<float> {};

    atlas::interactive::SystemFactory factory(config);
    const auto system = factory.create();
    EXPECT_NE(system->fluid()->state<atlas::FluidTemperatureState>(), nullptr);
}

TEST(InteractiveConfig, SystemFactoryKeepsTriangleMeshStorageAliveAcrossRebuilds) {
    Config config = make_config();
    Config::Collider collider;
    collider.unit.geometry.kind = Config::GeometryKind::triangle_mesh;
    collider.unit.geometry.triangles = {
        std::array<Config::Vec3, 3> {
            Config::Vec3 { 0.0f, 0.0f, 0.0f },
            Config::Vec3 { 1.0f, 0.0f, 0.0f },
            Config::Vec3 { 0.0f, 1.0f, 0.0f }
        }
    };
    config.colliders.push_back(collider);

    atlas::interactive::SystemFactory factory(config);
    auto first = factory.create();
    auto second = factory.create();
    first->update();
    second->update();
    EXPECT_EQ(first->step(), 1u);
    EXPECT_EQ(second->step(), 1u);
}

TEST(InteractiveServer, OwnsCreatesRoutesAndClosesSessions) {
    atlas::interactive::Server server;
    atlas::interactive::Request create;
    create.request_id = "create-1";
    create.command = atlas::interactive::Command::create;
    create.simulation = make_config();
    const auto created = server.handle(create);
    ASSERT_TRUE(created.success) << created.error;
    ASSERT_TRUE(created.session_id.has_value());
    EXPECT_EQ(created.request_id, "create-1");
    atlas::interactive::Request create_second = create;
    create_second.request_id = "create-2";
    const auto second = server.handle(create_second);
    ASSERT_TRUE(second.success) << second.error;
    ASSERT_TRUE(second.session_id.has_value());
    EXPECT_NE(created.session_id, second.session_id);
    EXPECT_EQ(server.session_count(), 2u);

    const std::uint64_t id = *created.session_id;
    EXPECT_TRUE(server.handle(request(atlas::interactive::Command::start, id)).success);
    server.update();
    const auto status = server.handle(request(atlas::interactive::Command::status, id));
    ASSERT_TRUE(status.status.has_value());
    EXPECT_EQ(status.status->step, 1u);
    const auto second_status = server.handle(
        request(atlas::interactive::Command::status, *second.session_id));
    ASSERT_TRUE(second_status.status.has_value());
    EXPECT_EQ(second_status.status->step, 0u);

    EXPECT_TRUE(server.handle(request(atlas::interactive::Command::close, id)).success);
    EXPECT_EQ(server.session_count(), 1u);
    EXPECT_FALSE(server.handle(request(atlas::interactive::Command::status, id)).success);
    EXPECT_TRUE(server.handle(
        request(atlas::interactive::Command::close, *second.session_id)).success);
    EXPECT_EQ(server.session_count(), 0u);

    atlas::interactive::Request shutdown;
    shutdown.command = atlas::interactive::Command::shutdown;
    EXPECT_TRUE(server.handle(shutdown).success);
    EXPECT_TRUE(server.shutdown_requested());
}

TEST(InteractiveServer, ReportsInvalidConfigurationAndSessionIds) {
    atlas::interactive::Server server;
    atlas::interactive::Request invalid_create;
    invalid_create.command = atlas::interactive::Command::create;
    invalid_create.simulation = make_config();
    invalid_create.simulation->fluid.particle_count = 5;
    EXPECT_FALSE(server.handle(invalid_create).success);

    const auto missing = server.handle(request(atlas::interactive::Command::status, 999));
    EXPECT_FALSE(missing.success);
    EXPECT_FALSE(missing.error.empty());
}

TEST(InteractiveTransport, MalformedJsonProducesAnErrorAndContinues) {
    std::istringstream input("{\n{\"request_id\":\"ok\",\"command\":\"shutdown\"}\n");
    std::ostringstream output;
    atlas::interactive::JsonLinesTransport transport(input, output);
    atlas::interactive::Request request;
    atlas::interactive::Response error;

    ASSERT_TRUE(transport.receive(request, error));
    EXPECT_EQ(request.request_id, "ok");
    EXPECT_EQ(request.command, atlas::interactive::Command::shutdown);
    EXPECT_NE(output.str().find("\"success\":false"), std::string::npos);
}

TEST(InteractiveSession, LifecycleStatisticsSaveAndRestartUseStoredConfig) {
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "atlas_interactive_json_session_test";
    std::filesystem::remove_all(output);

    atlas::interactive::OutputConfig output_config;
    output_config.csv_enabled = true;
    output_config.output_directory = output;
    output_config.csv_filename = "statistics.csv";
    atlas::interactive::Session session(make_observed_config(), output_config);

    EXPECT_EQ(session.status().state, atlas::interactive::SessionState::ready);
    session.start();
    session.update();
    EXPECT_EQ(session.status().step, 1u);
    EXPECT_EQ(session.statistics().sample_count(), 1u);
    session.pause();
    session.update();
    EXPECT_EQ(session.status().step, 1u);
    session.step(1);
    EXPECT_EQ(session.status().step, 2u);

    session.save(output / "snapshots");
    EXPECT_TRUE(std::filesystem::exists(output / "snapshots" / "time_step_2" / "fluid.bin"));
    session.restart();
    EXPECT_EQ(session.status().step, 0u);
    EXPECT_EQ(session.statistics().sample_count(), 0u);
    EXPECT_EQ(session.simulation_config().emitters.size(), 1u);
    std::filesystem::remove_all(output);
}

TEST(InteractiveScene, CoversEveryGeometryTypeAndTracksMovingPoses) {
    Config config = make_config();
    const std::array<Config::GeometryKind, 9> kinds = {
        Config::GeometryKind::box,
        Config::GeometryKind::circle,
        Config::GeometryKind::cylinder,
        Config::GeometryKind::plane,
        Config::GeometryKind::sphere,
        Config::GeometryKind::square,
        Config::GeometryKind::triangle,
        Config::GeometryKind::triangle_mesh,
        Config::GeometryKind::polygonal_prism
    };
    for (const Config::GeometryKind kind : kinds) {
        Config::Collider collider;
        collider.unit.geometry.kind = kind;
        collider.unit.velocity = Config::Vec3 { 1.0f, 0.0f, 0.0f };
        if (kind == Config::GeometryKind::triangle_mesh) {
            collider.unit.geometry.triangles = {
                std::array<Config::Vec3, 3> {
                    Config::Vec3 { 0.0f, 0.0f, 0.0f },
                    Config::Vec3 { 1.0f, 0.0f, 0.0f },
                    Config::Vec3 { 0.0f, 1.0f, 0.0f }
                }
            };
        }
        config.colliders.push_back(collider);
    }

    atlas::interactive::Session session(config);
    const auto before = session.scene_view();
    ASSERT_EQ(before.geometries.size(), kinds.size());
    for (std::size_t index = 0; index < kinds.size(); ++index) {
        ASSERT_NE(before.geometries[index].geometry, nullptr);
        EXPECT_EQ(before.geometries[index].geometry->kind, kinds[index]);
    }
    session.step();
    const auto after = session.scene_view();
    EXPECT_GT(after.geometries[0].sync.translation.x,
              before.geometries[0].sync.translation.x);
}

TEST(InteractiveScene, OscillatingColliderClosesReturnsAndRestartsAtItsInitialPose) {
    const auto config = decode_oscillating_config(
        R"({"axis":[0,0,1],"amplitude":1.57079632679,"period":2})");
    atlas::interactive::Session session(config);
    const atlas::Float3 tip(1.0f, 0.0f, 0.0f);
    const auto initial = session.scene_view().geometries[0].sync.sync_to_world(tip);
    EXPECT_FLOAT_EQ(initial.x, 2.0f);
    EXPECT_FLOAT_EQ(initial.y, 0.5f);

    session.step(4);
    const auto closed = session.scene_view().geometries[0].sync.sync_to_world(tip);
    EXPECT_NEAR(closed.x, 1.0f, 1e-5f);
    EXPECT_NEAR(closed.y, 1.5f, 1e-5f);
    session.pause();
    session.update();
    EXPECT_EQ(session.status().step, 4u);

    session.step(4);
    const auto reopened = session.scene_view().geometries[0].sync.sync_to_world(tip);
    EXPECT_NEAR(reopened.x, initial.x, 1e-5f);
    EXPECT_NEAR(reopened.y, initial.y, 1e-5f);

    session.step(2);
    session.restart();
    EXPECT_EQ(session.status().step, 0u);
    const auto restarted = session.scene_view().geometries[0].sync.sync_to_world(tip);
    EXPECT_NEAR(restarted.x, initial.x, 1e-5f);
    EXPECT_NEAR(restarted.y, initial.y, 1e-5f);
    session.step(4);
    const auto closed_again = session.scene_view().geometries[0].sync.sync_to_world(tip);
    EXPECT_NEAR(closed_again.x, closed.x, 1e-5f);
    EXPECT_NEAR(closed_again.y, closed.y, 1e-5f);
}

TEST(InteractiveProtocol, JsonRequestAndResponsePreserveCorrelationFields) {
    const auto request = atlas::interactive::JsonCodec::decode_request(
        R"({"request_id":"step-7","session_id":3,"command":"step","payload":{"step_count":4}})");
    EXPECT_EQ(request.request_id, "step-7");
    EXPECT_EQ(request.session_id, 3u);
    EXPECT_EQ(request.step_count, 4u);

    atlas::interactive::Response response;
    response.request_id = "step-7";
    response.success = true;
    response.session_id = 3;
    response.status = atlas::interactive::SessionStatus {};
    const std::string encoded = atlas::interactive::JsonCodec::encode_response(response);
    EXPECT_NE(encoded.find("\"request_id\":\"step-7\""), std::string::npos);
    EXPECT_NE(encoded.find("\"session_id\":3"), std::string::npos);
}

TEST(InteractiveProtocol, CreateRequestSeparatesSimulationAndOutputConfiguration) {
    const auto request = atlas::interactive::JsonCodec::decode_request(R"({
        "request_id":"create-8",
        "command":"create",
        "payload":{
            "config":{
                "dt":0.1,
                "fluid":{"buffer_size":4},
                "universe":{
                    "lower_corner":[-1,-1,-1],
                    "upper_corner":[1,1,1],
                    "cell_size":1
                }
            },
            "output":{
                "csv_enabled":true,
                "output_directory":"results",
                "csv_filename":"run.csv"
            }
        }
    })");

    ASSERT_TRUE(request.simulation.has_value());
    EXPECT_FLOAT_EQ(request.simulation->dt, 0.1f);
    EXPECT_TRUE(request.output.csv_enabled);
    EXPECT_EQ(request.output.output_directory, std::filesystem::path("results"));
    EXPECT_EQ(request.output.csv_filename, "run.csv");
}

TEST(InteractiveConfig, AngularOscillationInitialElapsedReachesCore) {
    for (const std::string smooth : { "true", "false" }) {
        SCOPED_TRACE(smooth);
        const auto config = decode_oscillating_config(
            R"({"axis":[0,0,1],"amplitude":1,"period":2,"elapsed":0.25,"smooth":)" + smooth + "}");
        const auto& configured_unit = config.colliders[0].unit;
        ASSERT_TRUE(configured_unit.angular_oscillation.has_value());
        EXPECT_FLOAT_EQ(configured_unit.angular_oscillation->elapsed, 0.25f);
        atlas::interactive::CoreFactory factory;
        const auto unit = factory.build(configured_unit);
        const float expected_angle = smooth == "true" ? 0.1464466094f : 0.25f;
        const auto direction = unit.sync().sync_dir_to_world(atlas::Float3(1.0f, 0.0f, 0.0f));
        EXPECT_NEAR(direction.x, std::cos(expected_angle), 1e-6f);
        EXPECT_NEAR(direction.y, std::sin(expected_angle), 1e-6f);
        ASSERT_TRUE(unit.angular_velocity().has_value());
        const float expected_speed = smooth == "true" ? 1.110720735f : 1.0f;
        EXPECT_NEAR(unit.angular_velocity()->z, expected_speed, 1e-6f);
    }
}

TEST(InteractiveScene, RestartRestoresTheConfiguredOscillationPhase) {
    for (const std::string smooth : { "true", "false" }) {
        SCOPED_TRACE(smooth);
        const auto config = decode_oscillating_config(
            R"({"axis":[0,0,1],"amplitude":1,"period":2,"elapsed":0.25,"smooth":)" + smooth + "}");
        atlas::interactive::Session session(config);
        const atlas::Float3 tip(1.0f, 0.0f, 0.0f);
        const float expected_angle = smooth == "true" ? 0.1464466094f : 0.25f;
        const auto initial = session.scene_view().geometries[0].sync.sync_to_world(tip);
        EXPECT_NEAR(initial.x, 1.0f + std::cos(expected_angle), 1e-6f);
        EXPECT_NEAR(initial.y, 0.5f + std::sin(expected_angle), 1e-6f);

        session.step(8);
        const auto after_cycle = session.scene_view().geometries[0].sync.sync_to_world(tip);
        EXPECT_NEAR(after_cycle.x, initial.x, 1e-5f);
        EXPECT_NEAR(after_cycle.y, initial.y, 1e-5f);
        session.step();
        const auto first_step = session.scene_view().geometries[0].sync.sync_to_world(tip);
        session.step(2);
        session.restart();
        EXPECT_EQ(session.status().step, 0u);
        const auto restarted = session.scene_view().geometries[0].sync.sync_to_world(tip);
        EXPECT_NEAR(restarted.x, initial.x, 1e-5f);
        EXPECT_NEAR(restarted.y, initial.y, 1e-5f);
        session.step();
        const auto first_step_again = session.scene_view().geometries[0].sync.sync_to_world(tip);
        EXPECT_NEAR(first_step_again.x, first_step.x, 1e-5f);
        EXPECT_NEAR(first_step_again.y, first_step.y, 1e-5f);
    }
}
