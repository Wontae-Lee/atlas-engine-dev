#pragma once

#include "config/output_config.h"
#include "config/simulation_config.h"
#include "config/system_factory.h"
#include "session/session_state.h"
#include "session/session_status.h"
#include "statistics/simulation_sample.h"
#include "statistics/simulation_statistics.h"
#include "view/simulation_render_view.h"
#include "view/simulation_scene_view.h"

#include <atlas/system/system.h>

#include <cstddef>
#include <filesystem>
#include <memory>

namespace atlas::interactive {

class CsvWriter;

class Session final {
public:
    explicit Session(SimulationConfig simulation, OutputConfig output = {});
    ~Session();

    void start();
    void pause();
    void step(std::size_t count = 1);
    void update();
    void save(const std::filesystem::path& path);
    void restart();

    bool running() const noexcept;
    SessionStatus status() const;
    const SimulationStatistics& statistics() const noexcept;
    SimulationRenderView render_view() const;
    SimulationSceneView scene_view() const;
    const SimulationConfig& simulation_config() const noexcept;

private:
    void initialize();
    void advance_once();
    void configure_csv();
    SimulationSample collect_sample() const;

    SystemFactory _factory;
    OutputConfig _output;
    atlas::SystemHostPtr _system;
    SessionState _state = SessionState::ready;
    SimulationStatistics _statistics;
    std::unique_ptr<CsvWriter> _csv_writer;
};

}
