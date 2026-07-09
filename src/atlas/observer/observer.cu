#include <atlas/observer/observer.h>

#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace atlas {

namespace {

    // A CSV column: its header and the value it reads out of a host copy of one
    // state's buffer. Absent states never build a column, so the file's header
    // says exactly which states the object carried.
    struct Column final {
        std::string header;
        std::vector<float> values;
    };

    template <typename StateT, typename Owner>
    void
    append_scalar_column(std::vector<Column>& columns,
                         const Owner& owner,
                         const char* header,
                         const std::size_t rows) {
        const auto* state = owner.template state<StateT>();

        if (state == nullptr) {
            return;
        }

        const HostBuffer<typename std::decay_t<decltype(state->data())>::value_type>
            values(state->data().begin(), state->data().end());

        if (values.size() < rows) {
            return;
        }

        Column column { header, {} };
        column.values.reserve(rows);

        for (std::size_t i = 0; i < rows; ++i) {
            column.values.push_back(static_cast<float>(values[i]));
        }

        columns.push_back(std::move(column));
    }

    template <typename StateT, typename Owner>
    void
    append_vector_columns(std::vector<Column>& columns,
                          const Owner& owner,
                          const char* header,
                          const std::size_t rows) {
        const auto* state = owner.template state<StateT>();

        if (state == nullptr) {
            return;
        }

        const HostBuffer<Float3> values(state->data().begin(), state->data().end());

        if (values.size() < rows) {
            return;
        }

        const char* axes[3] = { "_x", "_y", "_z" };

        for (int axis = 0; axis < 3; ++axis) {
            Column column { std::string(header) + axes[axis], {} };
            column.values.reserve(rows);

            for (std::size_t i = 0; i < rows; ++i) {
                column.values.push_back(values[i][static_cast<std::size_t>(axis)]);
            }

            columns.push_back(std::move(column));
        }
    }

    void
    write_csv(const std::filesystem::path& path,
              const char* index_header,
              const std::vector<Column>& columns,
              const std::size_t rows) {
        std::ofstream file(path, std::ios::out | std::ios::trunc);

        if (!file) {
            throw std::runtime_error("Observer::observe: cannot open " + path.string());
        }

        file << index_header;
        for (const auto& column : columns) {
            file << ',' << column.header;
        }
        file << '\n';

        for (std::size_t i = 0; i < rows; ++i) {
            file << i;
            for (const auto& column : columns) {
                file << ',' << column.values[i];
            }
            file << '\n';
        }
    }

}

Observer::Builder
Observer::builder() noexcept {
    return Builder {};
}


void
Observer::observe(const Fluid& fluid, const Universe& universe, const std::size_t step) const {
    if (_interval == 0 || step % _interval != 0) {
        return;
    }

    const std::filesystem::path data_directory = _output_directory / "data";

    std::filesystem::create_directories(data_directory);

    const std::string suffix = "_" + std::to_string(step) + ".csv";

    if (const std::size_t particles = fluid.particle_count(); particles > 0) {
        std::vector<Column> columns;

        append_vector_columns<FluidPositionState>(columns, fluid, "position", particles);
        append_vector_columns<FluidVelocityState>(columns, fluid, "velocity", particles);
        append_scalar_column<FluidSpeciesState>(columns, fluid, "species", particles);
        {
            const HostBuffer<int> active(fluid.active().begin(), fluid.active().end());

            if (active.size() >= particles) {
                Column column { "active", {} };
                column.values.reserve(particles);

                for (std::size_t i = 0; i < particles; ++i) {
                    column.values.push_back(static_cast<float>(active[i]));
                }

                columns.push_back(std::move(column));
            }
        }
        append_scalar_column<FluidTemperatureState>(columns, fluid, "temperature", particles);
        append_scalar_column<FluidTranslationalEnergyState>(columns, fluid, "translational_energy", particles);
        append_scalar_column<FluidRotationalEnergyState>(columns, fluid, "rotational_energy", particles);
        append_scalar_column<FluidVibrationalEnergyState>(columns, fluid, "vibrational_energy", particles);

        write_csv(data_directory / ("fluid" + suffix), "particle", columns, particles);
    }

    if (const auto cells = static_cast<std::size_t>(universe.cell_count()); cells > 0) {
        std::vector<Column> columns;

        append_scalar_column<UniverseTemperatureState>(columns, universe, "temperature", cells);
        append_vector_columns<UniverseBulkVelocityState>(columns, universe, "bulk_velocity", cells);
        append_vector_columns<UniverseFieldForceState>(columns, universe, "field_force", cells);
        append_vector_columns<UniverseGravityState>(columns, universe, "gravity", cells);
        append_scalar_column<UniverseMaxRelativeSpeedState>(columns, universe, "max_relative_speed", cells);
        append_scalar_column<UniverseMaxSigmaGState>(columns, universe, "max_sigma_g", cells);
        append_scalar_column<UniverseThermalEnergyState>(columns, universe, "thermal_energy", cells);
        append_scalar_column<UniverseNumberParticleState>(columns, universe, "number_particle", cells);
        append_scalar_column<UniverseCollisionCountState>(columns, universe, "collision_count", cells);
        append_scalar_column<UniverseKnudsenNumberState>(columns, universe, "knudsen_number", cells);
        append_scalar_column<UniverseAllocatedSolverState>(columns, universe, "allocated_solver", cells);

        write_csv(data_directory / ("universe" + suffix), "cell", columns, cells);
    }
}



Observer::Builder&
Observer::Builder::with_interval(const std::size_t interval) noexcept {
    _interval = interval;
    return *this;
}

Observer::Builder&
Observer::Builder::with_output_directory(std::filesystem::path output_directory) {
    _output_directory = std::move(output_directory);
    return *this;
}

Observer
Observer::Builder::build() const {
    Observer observer;

    observer._interval         = _interval;
    observer._output_directory = _output_directory;

    return observer;
}

atlas::host_shared_ptr<Observer>
Observer::Builder::make_host_shared() const {
    auto observer = build();

    return atlas::make_host_shared<Observer>(std::move(observer));
}

}
