#include "register.h"
#include "binding_types.h"
#include "state_access.h"

#include <atlas/codec/codec.h>
#include <atlas/generator/generator.h>
#include <atlas/observer/observer.h>
#include <atlas/solver/solver.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

void
register_system(nb::module_& m) {
    nb::class_<PySystem>(m, "System")
        .def("update", [](PySystem& system) { system.value.update(); },
             "Advance the simulation by one step.")
        .def("initialize_states", [](PySystem& system) { system.value.initialize_states(); })
        .def("emit", [](PySystem& system) { system.value.emit(); })
        .def("search", [](PySystem& system) { system.value.search(); })
        .def("allocate", [](PySystem& system) { system.value.allocate(); })
        .def("solve", [](PySystem& system) { system.value.solve(); })
        .def("advect", [](PySystem& system) { system.value.advect(); })
        .def("remove", [](PySystem& system) { system.value.remove(); })
        .def("save", [](const PySystem& system, const std::filesystem::path& directory) {
            system.value.save(directory);
        }, "directory"_a, "Serialize a per-step snapshot under the given directory.")
        .def_static("snapshot_directory_name", &System::snapshot_directory_name, "step"_a)
        .def_prop_ro("step", [](const PySystem& system) { return system.value.step(); })
        .def_prop_ro("dt", [](const PySystem& system) { return system.value.dt(); })
        .def_prop_ro("source_count", [](const PySystem& system) { return system.value.source_count(); })
        .def_prop_ro("solver_count", [](const PySystem& system) { return system.value.solver_count(); })
        .def_prop_ro("collider_count", [](const PySystem& system) { return system.value.collider_count(); })
        .def_prop_ro("sink_count", [](const PySystem& system) { return system.value.sink_count(); })
        .def_prop_ro("solvers", [](const PySystem& system) {
            const auto& solvers = system.value.solvers();
            return std::vector<SolverHostPtr>(solvers.begin(), solvers.end());
        })
        .def_prop_ro("codec", [](const PySystem& system) { return system.value.codec(); })
        .def_prop_ro("observer", [](const PySystem& system) { return system.value.observer(); })
        .def_prop_rw("particle_count",
            [](const PySystem& system) { return system.value.fluid()->particle_count(); },
            [](PySystem& system, const std::size_t count) { system.value.fluid()->set_particle_count(count); })
        .def_prop_ro("buffer_size", [](const PySystem& system) { return system.value.fluid()->buffer_size(); })
        .def_prop_ro("statistical_weight", [](const PySystem& system) { return system.value.fluid()->statistical_weight(); })
        .def_prop_ro("materials", [](const PySystem& system) { return system.value.fluid()->materials(); })
        .def_prop_ro("cell_count", [](const PySystem& system) { return system.value.universe()->cell_count(); })
        .def_prop_ro("lower_corner", [](const PySystem& system) { return system.value.universe()->lower_corner(); })
        .def_prop_ro("upper_corner", [](const PySystem& system) { return system.value.universe()->upper_corner(); })
        .def_prop_ro("grid_size", [](const PySystem& system) { return system.value.universe()->grid_size(); })
        .def_prop_ro("cell_size", [](const PySystem& system) { return system.value.universe()->cell_size(); })
        .def_prop_ro("cell_volume", [](const PySystem& system) { return system.value.universe()->cell_volume(); })
        .def_prop_ro("inverse_cell_size", [](const PySystem& system) { return system.value.universe()->inverse_cell_size(); })
        .def("positions", [](const PySystem& system) { return read_state(*system.value.fluid(), "position"); })
        .def("velocities", [](const PySystem& system) { return read_state(*system.value.fluid(), "velocity"); })
        .def("species", [](const PySystem& system) { return read_state(*system.value.fluid(), "species"); })
        .def("fluid_state", [](const PySystem& system, const std::string& name, const bool full) {
            return read_state(*system.value.fluid(), name, full);
        }, "name"_a, "full"_a = false)
        .def("set_fluid_state", [](PySystem& system, const std::string& name, nb::handle values, const std::size_t offset) {
            write_state(*system.value.fluid(), name, values, offset);
        }, "name"_a, "values"_a, "offset"_a = 0)
        .def("has_fluid_state", [](const PySystem& system, const std::string& name) {
            return has_state(*system.value.fluid(), name);
        }, "name"_a)
        .def("remove_fluid_state", [](PySystem& system, const std::string& name) {
            return remove_state(*system.value.fluid(), name);
        }, "name"_a)
        .def("reset_fluid_state", [](PySystem& system, const std::string& name) {
            reset_state(*system.value.fluid(), name);
        }, "name"_a)
        .def("universe_state", [](const PySystem& system, const std::string& name) {
            return read_state(*system.value.universe(), name);
        }, "name"_a)
        .def("set_universe_state", [](PySystem& system, const std::string& name, nb::handle values, const std::size_t offset) {
            write_state(*system.value.universe(), name, values, offset);
        }, "name"_a, "values"_a, "offset"_a = 0)
        .def("has_universe_state", [](const PySystem& system, const std::string& name) {
            return has_state(*system.value.universe(), name);
        }, "name"_a)
        .def("remove_universe_state", [](PySystem& system, const std::string& name) {
            return remove_state(*system.value.universe(), name);
        }, "name"_a)
        .def("reset_universe_state", [](PySystem& system, const std::string& name) {
            reset_state(*system.value.universe(), name);
        }, "name"_a)
        .def("active", [](const PySystem& system, const bool full) {
            const auto& fluid = *system.value.fluid();
            return numpy_copy(fluid.active(), full ? fluid.buffer_size() : fluid.particle_count());
        }, "full"_a = false)
        .def("set_active", [](PySystem& system, nb::handle values, const std::size_t offset) {
            write_active(*system.value.fluid(), values, offset);
        }, "values"_a, "offset"_a = 0)
        .def("compact", [](PySystem& system) { return system.value.fluid()->compact(); })
        .def("observe", [](const PySystem& system) {
            const auto& value = system.value;
            if (value.observer()) value.observer()->observe(*value.fluid(), *value.universe(), value.step());
        });

    nb::class_<Observer>(m, "Observer")
        .def_prop_ro("interval", &Observer::interval)
        .def_prop_ro("output_directory", &Observer::output_directory)
        .def_prop_ro("species_count", &Observer::species_count)
        .def("observe", &Observer::observe, "fluid"_a, "universe"_a, "step"_a)
        .def("observe", [](const Observer& observer, const PySystem& system) {
            observer.observe(*system.value.fluid(), *system.value.universe(), system.value.step());
        }, "system"_a)
        .def("resize_counters", &Observer::resize_counters, "source_count"_a, "sink_count"_a, "species_count"_a)
        .def("reset_counters", &Observer::reset_counters)
        .def("spawned", [](const Observer& observer) {
            const auto columns = observer.species_count();
            const auto rows = columns ? observer.spawned().size() / columns : 0;
            return numpy_copy(observer.spawned(), observer.spawned().size()).attr("reshape")(rows, columns);
        })
        .def("despawned", [](const Observer& observer) {
            const auto columns = observer.species_count();
            const auto rows = columns ? observer.despawned().size() / columns : 0;
            return numpy_copy(observer.despawned(), observer.despawned().size()).attr("reshape")(rows, columns);
        });

    m.def("observer", [](const std::size_t interval, const std::filesystem::path& output_directory) {
        return Observer::builder().with_interval(interval).with_output_directory(output_directory).make_host_shared();
    }, "interval"_a, "output_directory"_a,
       "A CSV observer sampling every `interval` steps into `output_directory`.");

    m.def(
        "build_system",
        [](std::unique_ptr<Fluid> fluid,
           std::unique_ptr<Universe> universe,
           const float dt,
           SolverHostPtr solver,
           std::optional<PySource> source,
           GeneratorHostPtr generator,
           std::vector<PyCollider> colliders,
           std::vector<PySink> sinks,
           CodecHostPtr codec,
           ObserverHostPtr observer,
           std::vector<SolverHostPtr> solvers,
           std::vector<std::pair<PySource, GeneratorHostPtr>> emitters) {
            if (source.has_value() != static_cast<bool>(generator)) {
                throw nb::value_error("source and generator must be provided together");
            }
            std::vector<std::shared_ptr<void>> policy_owners;
            if (solver) policy_owners.push_back(solver);
            for (const auto& entry : solvers) policy_owners.push_back(entry);
            if (source.has_value()) {
                policy_owners.push_back(source->value);
                policy_owners.push_back(generator);
            }
            for (const auto& [emitter_source, emitter_generator] : emitters) {
                policy_owners.push_back(emitter_source.value);
                policy_owners.push_back(emitter_generator);
            }

            System::Builder builder = System::builder();
            builder.with_fluid(std::move(fluid));
            builder.with_universe(std::move(universe));
            builder.with_dt(dt);

            if (solver) builder.with_solver(std::move(solver));
            for (auto& entry : solvers) builder.with_solver(std::move(entry));

            MeshOwners mesh_owners;
            if (source.has_value()) {
                builder.with_emitter(std::move(source->value), std::move(generator));
                mesh_owners.insert(mesh_owners.end(), source->mesh_owners.begin(), source->mesh_owners.end());
            }
            for (auto& [emitter_source, emitter_generator] : emitters) {
                builder.with_emitter(std::move(emitter_source.value), std::move(emitter_generator));
                mesh_owners.insert(mesh_owners.end(), emitter_source.mesh_owners.begin(), emitter_source.mesh_owners.end());
            }
            for (const PyCollider& collider : colliders) {
                builder.with_collider(collider.value);
                mesh_owners.insert(mesh_owners.end(), collider.mesh_owners.begin(), collider.mesh_owners.end());
            }
            for (const PySink& sink : sinks) {
                builder.with_sink(sink.value);
                mesh_owners.insert(mesh_owners.end(), sink.mesh_owners.begin(), sink.mesh_owners.end());
            }
            if (codec) builder.with_codec(std::move(codec));
            if (observer) builder.with_observer(std::move(observer));

            auto system = builder.build();
            return PySystem(std::move(system), std::move(mesh_owners), std::move(policy_owners));
        },
        "fluid"_a, "universe"_a, "dt"_a,
        "solver"_a = SolverHostPtr {},
        "source"_a = nb::none(),
        "generator"_a = GeneratorHostPtr {},
        "colliders"_a = std::vector<PyCollider> {},
        "sinks"_a = std::vector<PySink> {},
        "codec"_a = CodecHostPtr {},
        "observer"_a = ObserverHostPtr {},
        nb::kw_only(),
        "solvers"_a = std::vector<SolverHostPtr> {},
        "emitters"_a = std::vector<std::pair<PySource, GeneratorHostPtr>> {},
        "Assemble a System; singular solver/emitter arguments precede their list entries. "
        "Fluid and Universe are consumed.");
}

}
