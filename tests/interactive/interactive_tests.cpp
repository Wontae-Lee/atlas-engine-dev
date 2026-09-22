#include "protocol/command.h"
#include "server/server.h"
#include "session/session.h"
#include "session/session_config.h"
#include "session/session_state.h"

#include <atlas/atlas.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

atlas::SystemHostPtr
make_system() {
    auto fluid = atlas::Fluid::builder()
                     .with_buffer_size(1)
                     .with_particle_count(1)
                     .make_host_unique();

    const atlas::Float3 position(0.0f, 0.0f, 0.0f);
    const atlas::Float3 velocity(1.0f, 0.0f, 0.0f);
    atlas::copy_host_to_device(
        &position,
        fluid->state<atlas::FluidPositionState>()->data(),
        1);
    atlas::copy_host_to_device(
        &velocity,
        fluid->state<atlas::FluidVelocityState>()->data(),
        1);

    auto universe = atlas::Universe::builder()
                        .with_lower_corner(atlas::Float3(-2.0f))
                        .with_upper_corner(atlas::Float3(2.0f))
                        .with_cell_size(1.0f)
                        .make_host_unique();
    universe->emplace_state<atlas::UniverseNumberParticleState>(universe->cell_count());

    return atlas::System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_dt(0.25f)
        .make_host_unique();
}

atlas::SystemHostPtr
make_observed_system() {
    auto fluid = atlas::Fluid::builder()
                     .with_buffer_size(64)
                     .with_particle_count(0)
                     .make_host_unique();

    auto universe = atlas::Universe::builder()
                        .with_lower_corner(atlas::Float3(-2.0f))
                        .with_upper_corner(atlas::Float3(2.0f))
                        .with_cell_size(1.0f)
                        .make_host_unique();
    universe->emplace_state<atlas::UniverseNumberParticleState>(universe->cell_count());

    const auto sync = atlas::Sync::builder()
                          .with_rigid_pose(atlas::Float3(0.0f),
                                           atlas::Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
                          .make_host_shared();
    auto source_unit = atlas::Unit::builder()
                           .with_geometry(atlas::Geometry(
                               atlas::Box::builder()
                                   .with_lower_corner(atlas::Float3(-0.5f))
                                   .with_upper_corner(atlas::Float3(0.5f))
                                   .build()))
                           .with_sync(sync)
                           .build();
    auto sink_unit = atlas::Unit::builder()
                         .with_geometry(atlas::Geometry(
                             atlas::Box::builder()
                                 .with_lower_corner(atlas::Float3(-1.0f))
                                 .with_upper_corner(atlas::Float3(1.0f))
                                 .build()))
                         .with_sync(sync)
                         .build();

    auto source = atlas::make_host_shared<atlas::Source>(atlas::Source(
        atlas::VolumeSource::builder()
            .with_unit(std::move(source_unit))
            .with_spacing(0.5f)
            .build()));
    auto generator = atlas::make_host_shared<atlas::Generator>(atlas::Generator(
        atlas::UniformGenerator::builder()
            .with_species_ratios({ 1.0f })
            .with_species_numbers({ 0.0f })
            .with_min_value(0.0f)
            .with_max_value(0.0f)
            .build()));
    const atlas::Sink sink(atlas::VolumeSink::builder()
                               .with_unit(std::move(sink_unit))
                               .build());

    return atlas::System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_emitter(std::move(source), std::move(generator))
        .with_sink(sink)
        .with_dt(0.01f)
        .make_host_unique();
}

}

TEST(InteractiveSession, LifecycleUsesFactoryAndCentralizedStepping) {
    std::size_t factory_calls = 0;
    atlas::interactive::Session session([&factory_calls] {
        ++factory_calls;
        return make_system();
    });
    atlas::interactive::Server server(session);

    EXPECT_EQ(factory_calls, 1u);
    EXPECT_EQ(session.status().state, atlas::interactive::SessionState::ready);

    EXPECT_TRUE(server.handle({ atlas::interactive::Command::start }).success);
    session.update();
    EXPECT_EQ(session.status().step, 1u);
    EXPECT_EQ(session.statistics().sample_count(), 1u);

    EXPECT_TRUE(server.handle({ atlas::interactive::Command::pause }).success);
    session.update();
    EXPECT_EQ(session.status().step, 1u);

    atlas::interactive::Request step_request;
    step_request.command = atlas::interactive::Command::step;
    step_request.step_count = 10;
    EXPECT_TRUE(server.handle(step_request).success);
    EXPECT_EQ(session.status().step, 11u);
    EXPECT_EQ(session.statistics().sample_count(), 11u);

    EXPECT_TRUE(server.handle({ atlas::interactive::Command::restart }).success);
    EXPECT_EQ(factory_calls, 2u);
    EXPECT_EQ(session.status().state, atlas::interactive::SessionState::ready);
    EXPECT_EQ(session.status().step, 0u);
    EXPECT_EQ(session.statistics().sample_count(), 0u);

    EXPECT_TRUE(server.handle({ atlas::interactive::Command::close }).success);
    EXPECT_EQ(session.status().state, atlas::interactive::SessionState::empty);
}

TEST(InteractiveServer, ForwardsStatusAndShutdownCommands) {
    atlas::interactive::Session session(make_system);
    atlas::interactive::Server server(session);

    const auto status = server.handle({ atlas::interactive::Command::status });
    ASSERT_TRUE(status.success);
    ASSERT_TRUE(status.status.has_value());
    EXPECT_EQ(status.status->state, atlas::interactive::SessionState::ready);

    EXPECT_TRUE(server.handle({ atlas::interactive::Command::shutdown }).success);
    EXPECT_TRUE(server.shutdown_requested());
}

TEST(InteractiveSession, ExposesNonOwningNativeRenderBuffers) {
    atlas::interactive::Session session(make_system);

    const auto view = session.render_view();
    EXPECT_EQ(view.particle_count, 1u);
    EXPECT_NE(view.position.data, nullptr);
    EXPECT_EQ(view.position.bytes, sizeof(atlas::Float3));
    EXPECT_NE(view.velocity.data, nullptr);
    EXPECT_EQ(view.velocity.bytes, sizeof(atlas::Float3));
    EXPECT_NE(view.species.data, nullptr);
    EXPECT_EQ(view.species.bytes, sizeof(std::size_t));
}

TEST(InteractiveSession, AccumulatesPerSourceAndPerSinkStatistics) {
    atlas::interactive::Session session(make_observed_system);
    ASSERT_TRUE(session.handle({ atlas::interactive::Command::step }).success);

    const auto& first = session.statistics();
    ASSERT_EQ(first.current().source_spawned.size(), 1u);
    ASSERT_EQ(first.current().sink_removed.size(), 1u);
    EXPECT_GT(first.current().source_spawned[0], 0u);
    EXPECT_EQ(first.current().sink_removed[0], first.current().source_spawned[0]);
    EXPECT_EQ(first.total_source_spawned()[0], first.current().source_spawned[0]);
    EXPECT_EQ(first.total_sink_removed()[0], first.current().sink_removed[0]);
    EXPECT_EQ(session.status().particle_count, 0u);

    ASSERT_TRUE(session.handle({ atlas::interactive::Command::step }).success);
    const auto& second = session.statistics();
    EXPECT_EQ(second.total_source_spawned()[0], 2u * second.current().source_spawned[0]);
    EXPECT_EQ(second.total_sink_removed()[0], 2u * second.current().sink_removed[0]);
}

TEST(InteractiveSession, StreamsCsvAndFlushesBeforeSnapshotSave) {
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "atlas_interactive_session_test";
    std::filesystem::remove_all(output);

    atlas::interactive::SessionConfig config;
    config.csv_enabled = true;
    config.output_directory = output;
    config.csv_filename = "run.csv";

    atlas::interactive::Session session(make_observed_system, config);
    atlas::interactive::Request step_request;
    step_request.command = atlas::interactive::Command::step;
    step_request.step_count = 2;
    ASSERT_TRUE(session.handle(step_request).success);

    atlas::interactive::Request save_request;
    save_request.command = atlas::interactive::Command::save;
    save_request.path = output / "snapshots";
    ASSERT_TRUE(session.handle(save_request).success);

    std::ifstream csv(output / "run.csv");
    ASSERT_TRUE(csv.good());
    std::string header;
    std::string first;
    std::string second;
    ASSERT_TRUE(static_cast<bool>(std::getline(csv, header)));
    ASSERT_TRUE(static_cast<bool>(std::getline(csv, first)));
    ASSERT_TRUE(static_cast<bool>(std::getline(csv, second)));
    EXPECT_EQ(header,
              "step,time,particle_count,source_0_spawned,sink_0_removed,source_0_total,sink_0_total");
    EXPECT_FALSE(first.empty());
    EXPECT_FALSE(second.empty());
    EXPECT_TRUE(std::filesystem::exists(output / "snapshots" / "time_step_2" / "fluid.bin"));
    EXPECT_TRUE(std::filesystem::exists(output / "snapshots" / "time_step_2" / "universe.bin"));

    ASSERT_TRUE(session.handle({ atlas::interactive::Command::restart }).success);
    ASSERT_TRUE(session.handle({ atlas::interactive::Command::step }).success);
    ASSERT_TRUE(session.handle(save_request).success);

    std::ifstream restarted_csv(output / "run.csv");
    ASSERT_TRUE(restarted_csv.good());
    std::string restarted_header;
    std::string restarted_row;
    std::string unexpected_row;
    ASSERT_TRUE(static_cast<bool>(std::getline(restarted_csv, restarted_header)));
    ASSERT_TRUE(static_cast<bool>(std::getline(restarted_csv, restarted_row)));
    EXPECT_FALSE(static_cast<bool>(std::getline(restarted_csv, unexpected_row)));
    EXPECT_EQ(restarted_header, header);
    EXPECT_FALSE(restarted_row.empty());
    EXPECT_EQ(session.status().step, 1u);
    EXPECT_NEAR(session.status().simulation_time, 0.01, 1.0e-8);

    std::filesystem::remove_all(output);
}
