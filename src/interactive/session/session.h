/**
 * @file
 * @brief Declares ownership and control of one Atlas simulation session.
 */

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

/**
 * @brief Owns and advances one Atlas core simulation.
 *
 * Session controls lifecycle, persistence, and application-side reporting. It
 * exposes non-owning rendering views but has no dependency on OpenGL.
 */
class Session final {
public:
    /// Creates a ready session from simulation and output configuration.
    explicit Session(SimulationConfig simulation, OutputConfig output = {});
    ~Session();

    /// Enables automatic advancement by Server::update.
    void start();
    /// Stops automatic advancement without discarding state.
    void pause();
    /// Advances exactly count simulation steps.
    void step(std::size_t count = 1);
    /// Advances once only when the session is running.
    void update();
    /// Serializes the owned Atlas system to path.
    void save(const std::filesystem::path& path);
    /// Recreates the system from its original configuration.
    void restart();

    /// Reports whether automatic advancement is enabled.
    bool running() const noexcept;
    /// Returns a current externally reportable status snapshot.
    SessionStatus status() const;
    /// Returns accumulated application-side statistics.
    const SimulationStatistics& statistics() const noexcept;
    /// Returns non-owning views of the live particle buffers.
    SimulationRenderView render_view() const;
    /// Returns particle, geometry, and domain views for rendering.
    SimulationSceneView scene_view() const;
    /// Returns the immutable configuration used to build the system.
    const SimulationConfig& simulation_config() const noexcept;

private:
    void initialize();
    void advance_once();
    void configure_csv();
    SimulationSample collect_sample() const;

    SystemFactory _factory; ///< Reusable source for initial and restarted systems.
    OutputConfig _output; ///< Application reporting policy.
    atlas::SystemHostPtr _system; ///< Owned core simulation.
    SessionState _state = SessionState::ready; ///< Interactive lifecycle state.
    SimulationStatistics _statistics; ///< Statistics accumulated since initialization.
    std::unique_ptr<CsvWriter> _csv_writer; ///< Optional application-level report sink.
};

}
