#include <atlas/observer/observer.h>

#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace atlas {

namespace {

    /**
     * @brief One materialized CSV column: a header cell plus the values beneath it.
     *
     * Built on the host from a host copy of one state's device buffer. Absent states
     * never produce a @c Column, so the emitted header row names exactly the states the
     * serialized object actually carried — the file is self-describing.
     */
    struct Column final {
        std::string header;        ///< The column's header-row label.
        std::vector<float> values; ///< One value per output row, already cast to float.
    };

    /**
     * @brief Appends one float column read from a scalar state, if that state is present.
     *
     * Looks up @p StateT on @p owner; if the state is not registered, or its host copy is
     * shorter than @p rows, no column is appended (the snapshot silently omits it). The
     * state's element type may be any arithmetic type — it is copied to the host and each
     * of the first @p rows entries is cast to @c float for the CSV.
     *
     * @tparam StateT The concrete scalar state leaf to read (e.g. @c FluidSpeciesState).
     * @tparam Owner The owning container (@c Fluid or @c Universe) exposing @c state<>().
     * @param columns Destination column list to append to.
     * @param owner The object whose state buffer is serialized.
     * @param header The header label for the new column.
     * @param rows Number of leading rows to emit (live particles or cells).
     */
    template <typename StateT, typename Owner>
    void
    append_scalar_column(std::vector<Column>& columns,
                         const Owner& owner,
                         const char* header,
                         const std::size_t rows) {
        const auto* state = owner.template state<StateT>();

        // An unregistered state contributes no column rather than an error.
        if (state == nullptr) {
            return;
        }

        // Pull the device buffer to the host once; the element type is whatever the
        // state stores (int, float, ...), deduced from data()'s value_type.
        const HostBuffer<typename std::decay_t<decltype(state->data())>::value_type>
            values(state->data().begin(), state->data().end());

        // Guard against a state shorter than the requested row count (e.g. a partially
        // sized buffer) so the per-row loop below never reads out of bounds.
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

    /**
     * @brief Appends three float columns (_x/_y/_z) from a @c Float3 vector state.
     *
     * The vector-valued twin of @c append_scalar_column: reads a state of @c Float3 to
     * the host and expands it into three columns named @p header + "_x"/"_y"/"_z". Does
     * nothing if the state is absent or its host copy is shorter than @p rows.
     *
     * @tparam StateT The concrete @c Float3-valued state leaf (e.g. @c FluidVelocityState).
     * @tparam Owner The owning container (@c Fluid or @c Universe) exposing @c state<>().
     * @param columns Destination column list to append to.
     * @param owner The object whose state buffer is serialized.
     * @param header The base label; each axis suffix is appended to it.
     * @param rows Number of leading rows to emit (live particles or cells).
     */
    template <typename StateT, typename Owner>
    void
    append_vector_columns(std::vector<Column>& columns,
                          const Owner& owner,
                          const char* header,
                          const std::size_t rows) {
        const auto* state = owner.template state<StateT>();

        // An unregistered state contributes no columns rather than an error.
        if (state == nullptr) {
            return;
        }

        const HostBuffer<Float3> values(state->data().begin(), state->data().end());

        // Same short-buffer guard as the scalar path; protects all three axis loops.
        if (values.size() < rows) {
            return;
        }

        const char* axes[3] = { "_x", "_y", "_z" };

        // One pass per component; each yields an independent column so downstream
        // tools can plot x/y/z separately.
        for (int axis = 0; axis < 3; ++axis) {
            Column column { std::string(header) + axes[axis], {} };
            column.values.reserve(rows);

            for (std::size_t i = 0; i < rows; ++i) {
                column.values.push_back(values[i][static_cast<std::size_t>(axis)]);
            }

            columns.push_back(std::move(column));
        }
    }

    /**
     * @brief Writes the assembled columns to a CSV file, truncating any existing file.
     *
     * Emits a header row of @p index_header followed by each column's header, then one
     * row per index @c i in @c [0, rows): the index value itself followed by every
     * column's @c values[i]. All columns are assumed to hold at least @p rows entries,
     * which the @c append_* helpers guarantee by refusing to build short columns.
     *
     * @param path Destination file path (its parent directory must already exist).
     * @param index_header Label for the leading index column (e.g. "particle", "cell").
     * @param columns The data columns, in output order.
     * @param rows Number of data rows to write.
     * @throws std::runtime_error if the file cannot be opened for writing.
     */
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

    /**
     * @brief Serializes a per-entity/per-species counter buffer to CSV.
     *
     * The @p counter is the flat row-major [entity][species] tally that @c System fills
     * on the device (spawns per source, or despawns per sink). This transposes the flat
     * layout into one @c "species_<k>" column per species and one row per entity, then
     * writes it via @c write_csv. Returns without writing if @p species_count is 0,
     * which also avoids a divide-by-zero when deriving the row count.
     *
     * @param path Destination file path.
     * @param index_header Label for the leading index column ("source" or "sink").
     * @param counter The device-side counter buffer to serialize.
     * @param species_count Column stride; number of species per entity row.
     * @throws std::runtime_error if the file cannot be opened (propagated from write_csv).
     */
    void
    write_counter_csv(const std::filesystem::path& path,
                      const char* index_header,
                      const DeviceBuffer<int>& counter,
                      const std::size_t species_count) {
        // Guard the divide below and skip empty-schema output.
        if (species_count == 0) {
            return;
        }

        const HostBuffer<int> values(counter.begin(), counter.end());
        const std::size_t rows = values.size() / species_count; // entities = length / stride

        std::vector<Column> columns;
        columns.reserve(species_count);

        for (std::size_t species = 0; species < species_count; ++species) {
            Column column { "species_" + std::to_string(species), {} };
            column.values.reserve(rows);

            for (std::size_t row = 0; row < rows; ++row) {
                column.values.push_back(static_cast<float>(values[row * species_count + species]));
            }

            columns.push_back(std::move(column));
        }

        write_csv(path, index_header, columns, rows);
    }

}

Observer::Builder
Observer::builder() noexcept {
    return Builder {};
}

void
Observer::observe(const Fluid& fluid, const Universe& universe, const std::size_t step) const {
    // Interval gate: 0 disables output entirely, otherwise only act on multiples.
    if (_interval == 0 || step % _interval != 0) {
        return;
    }

    const std::filesystem::path data_directory = _output_directory / "data";

    std::filesystem::create_directories(data_directory);

    const std::string suffix = "_" + std::to_string(step) + ".csv";

    // Fluid snapshot: one row per live particle. Skipped when there are none.
    if (const std::size_t particles = fluid.particle_count(); particles > 0) {
        std::vector<Column> columns;

        append_vector_columns<FluidPositionState>(columns, fluid, "position", particles);
        append_vector_columns<FluidVelocityState>(columns, fluid, "velocity", particles);
        append_scalar_column<FluidSpeciesState>(columns, fluid, "species", particles);
        {
            // The survivor flag is stored on the fluid itself, not as a tagged state, so
            // it is copied and appended by hand rather than through append_scalar_column.
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

    // Universe snapshot: one row per grid cell. Skipped when the grid is empty.
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

    // Spawn/despawn tallies: emitted only when the counters were actually sized.
    if (!_spawned.empty()) {
        write_counter_csv(data_directory / ("source" + suffix), "source", _spawned, _species_count);
    }

    if (!_despawned.empty()) {
        write_counter_csv(data_directory / ("sink" + suffix), "sink", _despawned, _species_count);
    }
}

void
Observer::resize_counters(const std::size_t source_count,
                          const std::size_t sink_count,
                          const std::size_t species_count) {
    _species_count = species_count;

    // assign() both resizes to the row-major [entity][species] length and zero-fills.
    _spawned.assign(source_count * species_count, 0);
    _despawned.assign(sink_count * species_count, 0);
}

void
Observer::reset_counters() {
    // Zero in place on the device; sizes stay as resize_counters() set them.
    atlas::parallel_fill<ExecutionPolicy::device>(_spawned.begin(), _spawned.end(), 0);
    atlas::parallel_fill<ExecutionPolicy::device>(_despawned.begin(), _despawned.end(), 0);
}

Observer::Builder&
Observer::Builder::with_interval(const std::size_t interval) noexcept {
    _interval = interval; // stashed; copied into the Observer by build()
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

    // Builder is a friend, so it writes the private fields directly. Counters are left
    // empty here and sized later by Observer::resize_counters().
    observer._interval         = _interval;
    observer._output_directory = _output_directory;

    return observer;
}

atlas::host_shared_ptr<Observer>
Observer::Builder::make_host_shared() const {
    auto observer = build();

    // Move the built value into the shared owner; Observer is move-only.
    return atlas::make_host_shared<Observer>(std::move(observer));
}

}
