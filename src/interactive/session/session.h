#pragma once

#include "protocol/request.h"
#include "protocol/response.h"
#include "session/session_config.h"
#include "session/session_state.h"
#include "session/session_status.h"
#include "session/system_factory.h"
#include "statistics/simulation_sample.h"
#include "statistics/simulation_statistics.h"
#include "view/simulation_render_view.h"

#include <atlas/system/system.h>

#include <cstddef>
#include <filesystem>
#include <memory>

namespace atlas::interactive {

class CsvWriter;

class Session final {
public:
    explicit Session(SystemFactory factory, SessionConfig config = {});
    ~Session();

    Response handle(const Request& request);
    void update();

    SessionStatus status() const;
    const SimulationStatistics& statistics() const noexcept;
    SimulationRenderView render_view() const;
    bool shutdown_requested() const noexcept;

private:
    void initialize();
    void start();
    void pause();
    void step(std::size_t count);
    void advance_once();
    void save(const std::filesystem::path& path);
    void restart();
    void close();
    void configure_csv();
    SimulationSample collect_sample() const;

    SystemFactory _factory;
    SessionConfig _config;
    atlas::SystemHostPtr _system;
    SessionState _state = SessionState::empty;
    SimulationStatistics _statistics;
    std::unique_ptr<CsvWriter> _csv_writer;
    bool _shutdown_requested = false;
};

}
