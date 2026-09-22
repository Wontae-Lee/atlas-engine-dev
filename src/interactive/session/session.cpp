#include "session/session.h"

#include "output/csv_writer.h"

#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace atlas::interactive {

namespace {

template <typename State, typename Value>
SimulationBufferView
required_buffer(const Fluid& fluid, const std::size_t count) {
    const State* state = fluid.state<State>();
    if (state == nullptr) throw std::logic_error("Fluid is missing a required render state.");
    return { atlas::raw_pointer_cast(state->data().data()), count * sizeof(Value) };
}

template <typename State, typename Value>
std::optional<SimulationBufferView>
optional_buffer(const Fluid& fluid, const std::size_t count) {
    const State* state = fluid.state<State>();
    if (state == nullptr) return std::nullopt;
    return SimulationBufferView {
        atlas::raw_pointer_cast(state->data().data()),
        count * sizeof(Value)
    };
}

}

Session::Session(SystemFactory factory, SessionConfig config)
    : _factory(std::move(factory))
    , _config(std::move(config)) {
    if (!_factory) throw std::invalid_argument("Session requires a SystemFactory.");
    initialize();
}

Session::~Session() = default;

Response
Session::handle(const Request& request) {
    try {
        switch (request.command) {
        case Command::start:
            start();
            return { true, "started", status() };
        case Command::pause:
            pause();
            return { true, "paused", status() };
        case Command::step:
            step(request.step_count);
            return { true, "stepped", status() };
        case Command::status:
            return { true, "status", status() };
        case Command::save:
            save(request.path);
            return { true, "saved", status() };
        case Command::restart:
            restart();
            return { true, "restarted", status() };
        case Command::close:
            close();
            return { true, "closed", status() };
        case Command::shutdown:
            _shutdown_requested = true;
            return { true, "shutdown requested", status() };
        }
    } catch (const std::exception& error) {
        return { false, error.what(), status() };
    }
    return { false, "unknown command", status() };
}

void
Session::update() {
    if (_state == SessionState::running) advance_once();
}

SessionStatus
Session::status() const {
    SessionStatus result;
    result.state = _state;
    if (!_system) return result;

    result.step = _system->step();
    result.simulation_time = static_cast<double>(_system->step()) * _system->dt();
    result.particle_count = _system->fluid()->particle_count();
    result.source_count = _system->source_count();
    result.sink_count = _system->sink_count();
    return result;
}

const SimulationStatistics&
Session::statistics() const noexcept {
    return _statistics;
}

SimulationRenderView
Session::render_view() const {
    if (!_system || !_system->fluid()) throw std::logic_error("Session has no renderable System.");

    const Fluid& fluid = *_system->fluid();
    const std::size_t count = fluid.particle_count();
    return {
        count,
        required_buffer<FluidPositionState, Float3>(fluid, count),
        required_buffer<FluidVelocityState, Float3>(fluid, count),
        required_buffer<FluidSpeciesState, std::size_t>(fluid, count),
        optional_buffer<FluidTemperatureState, float>(fluid, count),
        optional_buffer<FluidTranslationalEnergyState, float>(fluid, count),
        optional_buffer<FluidRotationalEnergyState, float>(fluid, count),
        optional_buffer<FluidVibrationalEnergyState, float>(fluid, count)
    };
}

bool
Session::shutdown_requested() const noexcept {
    return _shutdown_requested;
}

void
Session::initialize() {
    _system = _factory();
    if (!_system) throw std::runtime_error("SystemFactory returned a null System.");
    _statistics.reset();
    _state = SessionState::ready;
    configure_csv();
}

void
Session::start() {
    if (!_system) throw std::logic_error("Session must be initialized before it starts.");
    _state = SessionState::running;
}

void
Session::pause() {
    if (!_system) throw std::logic_error("Session must be initialized before it pauses.");
    _state = SessionState::paused;
}

void
Session::step(const std::size_t count) {
    if (!_system) throw std::logic_error("Session must be initialized before it steps.");
    for (std::size_t index = 0; index < count; ++index) advance_once();
}

void
Session::advance_once() {
    if (!_system) throw std::logic_error("Session must be initialized before it advances.");
    _system->update();
    const SimulationSample sample = collect_sample();
    _statistics.update(sample);
    if (_csv_writer) _csv_writer->append(sample, _statistics);
}

void
Session::save(const std::filesystem::path& path) {
    if (!_system) throw std::logic_error("Session must be initialized before it saves.");
    if (path.empty()) throw std::invalid_argument("Save requires an output directory.");
    if (_csv_writer) _csv_writer->flush();
    _system->save(path);
}

void
Session::restart() {
    _csv_writer.reset();
    _system.reset();
    _state = SessionState::empty;
    initialize();
}

void
Session::close() {
    if (_csv_writer) _csv_writer->flush();
    _csv_writer.reset();
    _system.reset();
    _statistics.reset();
    _state = SessionState::empty;
}

void
Session::configure_csv() {
    _csv_writer.reset();
    if (!_config.csv_enabled) return;
    if (_config.csv_filename.empty()) throw std::invalid_argument("CSV file name cannot be empty.");
    _csv_writer = std::make_unique<CsvWriter>(
        _config.output_directory / _config.csv_filename,
        _system->source_count(),
        _system->sink_count());
}

SimulationSample
Session::collect_sample() const {
    SimulationSample sample;
    sample.step = _system->step();
    sample.simulation_time = static_cast<double>(_system->step()) * _system->dt();
    sample.particle_count = _system->fluid()->particle_count();

    const auto& spawned = _system->source_spawned_last_step();
    sample.source_spawned.reserve(spawned.size());
    for (const int value : spawned) sample.source_spawned.push_back(static_cast<std::size_t>(std::max(value, 0)));

    const auto removed = _system->sink_removed_last_step();
    sample.sink_removed.reserve(removed.size());
    for (const int value : removed) sample.sink_removed.push_back(static_cast<std::size_t>(std::max(value, 0)));
    return sample;
}

}
