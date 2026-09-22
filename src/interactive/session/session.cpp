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

Session::Session(SimulationConfig simulation, OutputConfig output)
    : _factory(std::move(simulation))
    , _output(std::move(output)) {
    initialize();
}

Session::~Session() = default;

void
Session::update() {
    if (_state == SessionState::running) advance_once();
}

bool
Session::running() const noexcept {
    return _state == SessionState::running;
}

SessionStatus
Session::status() const {
    SessionStatus result;
    result.state = _state;
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

SimulationSceneView
Session::scene_view() const {
    SimulationSceneView result;
    result.particles = render_view();
    result.lower_corner = _system->universe()->lower_corner();
    result.upper_corner = _system->universe()->upper_corner();

    const SimulationConfig& config = _factory.config();
    const auto source_syncs = _system->source_syncs();
    const auto collider_syncs = _system->collider_syncs();
    const auto sink_syncs = _system->sink_syncs();
    if (source_syncs.size() != config.emitters.size()
        || collider_syncs.size() != config.colliders.size()
        || sink_syncs.size() != config.sinks.size()) {
        throw std::logic_error("Configured geometry count does not match the live System.");
    }
    result.geometries.reserve(source_syncs.size() + collider_syncs.size() + sink_syncs.size());
    for (std::size_t index = 0; index < source_syncs.size(); ++index) {
        result.geometries.push_back({ &config.emitters[index].source.unit.geometry,
                                      source_syncs[index],
                                      GeometryRenderView::Role::source });
    }
    for (std::size_t index = 0; index < collider_syncs.size(); ++index) {
        result.geometries.push_back({ &config.colliders[index].unit.geometry,
                                      collider_syncs[index],
                                      GeometryRenderView::Role::collider });
    }
    for (std::size_t index = 0; index < sink_syncs.size(); ++index) {
        result.geometries.push_back({ &config.sinks[index].unit.geometry,
                                      sink_syncs[index],
                                      GeometryRenderView::Role::sink });
    }
    return result;
}

void
Session::initialize() {
    _system = _factory.create();
    _statistics.reset();
    _state = SessionState::ready;
    configure_csv();
}

void
Session::start() {
    _state = SessionState::running;
}

void
Session::pause() {
    _state = SessionState::paused;
}

void
Session::step(const std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) advance_once();
}

void
Session::advance_once() {
    _system->update();
    const SimulationSample sample = collect_sample();
    _statistics.update(sample);
    if (_csv_writer) _csv_writer->append(sample, _statistics);
}

void
Session::save(const std::filesystem::path& path) {
    if (path.empty()) throw std::invalid_argument("Save requires an output directory.");
    if (_csv_writer) _csv_writer->flush();
    _system->save(path);
}

void
Session::restart() {
    _csv_writer.reset();
    _system.reset();
    initialize();
}

void
Session::configure_csv() {
    _csv_writer.reset();
    if (!_output.csv_enabled) return;
    if (_output.csv_filename.empty()) throw std::invalid_argument("CSV file name cannot be empty.");
    _csv_writer = std::make_unique<CsvWriter>(
        _output.output_directory / _output.csv_filename,
        _system->source_count(),
        _system->sink_count());
}

const SimulationConfig&
Session::simulation_config() const noexcept {
    return _factory.config();
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
