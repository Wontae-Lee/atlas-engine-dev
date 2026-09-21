#pragma once

#include "../detail/ownership.h"

#include <atlas/codec/codec.h>
#include <atlas/generator/generator.h>
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
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_system(nb::module_& m) {
    auto type = nb::class_<PySystem>(m, "System")
        .def(
            "update",
            [](PySystem& system) { system.value.update(); },
            "Advance the simulation by one step.")
        .def("initialize_states", [](PySystem& system) { system.value.initialize_states(); })
        .def("emit", [](PySystem& system) { system.value.emit(); })
        .def("search", [](PySystem& system) { system.value.search(); })
        .def("allocate", [](PySystem& system) { system.value.allocate(); })
        .def("solve", [](PySystem& system) { system.value.solve(); })
        .def("advect", [](PySystem& system) { system.value.advect(); })
        .def("remove", [](PySystem& system) { system.value.remove(); })
        .def(
            "save",
            [](const PySystem& system, const std::filesystem::path& directory) {
                system.value.save(directory);
            },
            "directory"_a,
            "Serialize a per-step snapshot under the given directory.")
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
        .def_prop_ro(
            "fluid",
            [](PySystem& system) -> Fluid& { return *system.value.fluid(); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "universe",
            [](PySystem& system) -> Universe& { return *system.value.universe(); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "searcher",
            [](PySystem& system) -> SpatialHashingSearcher& { return *system.value.searcher(); },
            nb::rv_policy::reference_internal);

    type.def("__init__", [](PySystem* value, std::unique_ptr<Fluid> fluid,
           std::unique_ptr<Universe>
               universe,
           const float dt,
           SolverHostPtr solver,
           std::optional<PySource>
               source,
           GeneratorHostPtr generator,
           std::vector<PyCollider>
               colliders,
           std::vector<PySink>
               sinks,
           CodecHostPtr codec,
           std::vector<SolverHostPtr>
               solvers,
           std::vector<std::pair<PySource, GeneratorHostPtr>>
               emitters) {
            if (source.has_value() != static_cast<bool>(generator)) {
                throw nb::value_error("source and generator must be provided together");
            }

            auto owned_fluid = atlas::make_host_unique<Fluid>(std::move(*fluid));
            auto owned_universe = atlas::make_host_unique<Universe>(std::move(*universe));
            fluid.reset();
            universe.reset();

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
            builder.with_fluid(std::move(owned_fluid));
            builder.with_universe(std::move(owned_universe));
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

            auto system = builder.build();
            new (value) PySystem(std::move(system), std::move(mesh_owners), std::move(policy_owners));
        },
        "fluid"_a,
        "universe"_a,
        "dt"_a,
        "solver"_a    = SolverHostPtr {},
        "source"_a    = nb::none(),
        "generator"_a = GeneratorHostPtr {},
        "colliders"_a = std::vector<PyCollider> {},
        "sinks"_a     = std::vector<PySink> {},
        "codec"_a     = CodecHostPtr {},
        nb::kw_only(),
        "solvers"_a  = std::vector<SolverHostPtr> {},
        "emitters"_a = std::vector<std::pair<PySource,
        GeneratorHostPtr>> {},
        "Assemble a System; singular solver/emitter arguments precede their list entries. "
        "Fluid and Universe are consumed.");
}

}
