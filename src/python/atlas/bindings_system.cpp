#include "register.h"

#include <atlas/codec/codec.h>
#include <atlas/collider/collider.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generator.h>
#include <atlas/sink/sink.h>
#include <atlas/solver/solver.h>
#include <atlas/source/source.h>
#include <atlas/system/system.h>
#include <atlas/universe/universe.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <filesystem>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// The capstone: assemble every subsystem the other groups produce into a runnable
// System, and expose the step loop plus read-only handles onto the live state.
namespace atlas::python {

void
register_system(nb::module_& m) {
    nb::class_<System>(m, "System")
        .def("update", &System::update, "Advance the simulation by one step.")
        .def("save", &System::save, "directory"_a,
             "Serialize a per-step snapshot under the given directory.")
        .def_prop_ro("step", &System::step)
        .def_prop_ro("dt", &System::dt)
        // Scalar read-backs onto the owned state. Returning the Fluid/Universe
        // objects themselves is unsafe here: they are the same C++ instances that
        // build_system consumed, which nanobind has marked relinquished. Reading
        // the counts through the System sidesteps that.
        .def_prop_ro(
            "particle_count",
            [](const System& s) { return s.fluid() ? s.fluid()->particle_count() : std::size_t {0}; })
        .def_prop_ro(
            "buffer_size",
            [](const System& s) { return s.fluid() ? s.fluid()->buffer_size() : std::size_t {0}; })
        .def_prop_ro(
            "cell_count",
            [](const System& s) { return s.universe() ? s.universe()->cell_count() : 0; });

    // A single assembly entry point. Fluid and Universe are unique_ptr owners and
    // are moved in; the policy objects are shared/value handles the other groups
    // hand out. Optional subsystems default to empty and are simply not wired.
    m.def(
        "build_system",
        [](std::unique_ptr<Fluid> fluid,
           std::unique_ptr<Universe> universe,
           const float dt,
           SolverHostPtr solver,
           SourceHostPtr source,
           GeneratorHostPtr generator,
           std::vector<Collider> colliders,
           std::vector<Sink> sinks,
           CodecHostPtr codec) {
            System::Builder builder = System::builder();
            builder.with_fluid(std::move(fluid));
            builder.with_universe(std::move(universe));
            builder.with_dt(dt);

            if (solver) {
                builder.with_solver(std::move(solver));
            }
            // The emitter is a source/generator pair; wire it only when both are given.
            if (source && generator) {
                builder.with_emitter(std::move(source), std::move(generator));
            }
            for (const Collider& collider : colliders) {
                builder.with_collider(collider);
            }
            for (const Sink& sink : sinks) {
                builder.with_sink(sink);
            }
            if (codec) {
                builder.with_codec(std::move(codec));
            }

            return builder.build();
        },
        "fluid"_a, "universe"_a, "dt"_a,
        "solver"_a    = SolverHostPtr {},
        "source"_a    = SourceHostPtr {},
        "generator"_a = GeneratorHostPtr {},
        "colliders"_a = std::vector<Collider> {},
        "sinks"_a     = std::vector<Sink> {},
        "codec"_a     = CodecHostPtr {},
        "Assemble a runnable System from its subsystems (fluid and universe are consumed).");
}

}
