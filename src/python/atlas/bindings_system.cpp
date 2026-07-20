#include "register.h"
#include "binding_types.h"

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/collider/collider.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/observer/observer.h>
#include <atlas/sink/sink.h>
#include <atlas/solver/solver.h>
#include <atlas/source/source.h>
#include <atlas/system/system.h>
#include <atlas/universe/universe.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>

namespace {

// Copy the live prefix of a per-particle Float3 column to the host and hand it
// back as an owned (N, 3) float32 numpy array. Returns an empty (0, 3) array
// when the fluid or the requested column is absent.
template <typename StateT>
nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
column_to_numpy(const atlas::python::PySystem& python_system) {
    namespace nb = nanobind;

    const atlas::System& system = python_system.value;
    const atlas::Fluid* fluid = system.fluid().get();
    const auto* state         = fluid ? fluid->state<StateT>() : nullptr;
    const std::size_t n       = fluid ? fluid->particle_count() : 0;

    auto* buffer = new float[3 * n];
    if (state != nullptr) {
        const auto& device = state->data();
        const atlas::HostBuffer<atlas::Float3> host(device.begin(), device.end());
        for (std::size_t i = 0; i < n; ++i) {
            buffer[3 * i + 0] = host[i].x;
            buffer[3 * i + 1] = host[i].y;
            buffer[3 * i + 2] = host[i].z;
        }
    }

    nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::shape<-1, 3>>(buffer, { n, 3 }, owner);
}

}

namespace nb = nanobind;
using namespace nb::literals;

// The capstone: assemble every subsystem the other groups produce into a runnable
// System, and expose the step loop plus read-only handles onto the live state.
namespace atlas::python {

void
register_system(nb::module_& m) {
    nb::class_<PySystem>(m, "System")
        .def("update", [](PySystem& system) { system.value.update(); },
             "Advance the simulation by one step.")
        .def("save", [](const PySystem& system, const std::filesystem::path& directory) {
                 system.value.save(directory);
             }, "directory"_a,
             "Serialize a per-step snapshot under the given directory.")
        .def_prop_ro("step", [](const PySystem& system) { return system.value.step(); })
        .def_prop_ro("dt", [](const PySystem& system) { return system.value.dt(); })
        // Scalar read-backs onto the owned state. Returning the Fluid/Universe
        // objects themselves is unsafe here: they are the same C++ instances that
        // build_system consumed, which nanobind has marked relinquished. Reading
        // the counts through the System sidesteps that.
        .def_prop_ro(
            "particle_count",
            [](const PySystem& system) {
                const System& s = system.value;
                return s.fluid() ? s.fluid()->particle_count() : std::size_t {0};
            })
        .def_prop_ro(
            "buffer_size",
            [](const PySystem& system) {
                const System& s = system.value;
                return s.fluid() ? s.fluid()->buffer_size() : std::size_t {0};
            })
        .def_prop_ro(
            "cell_count",
            [](const PySystem& system) {
                const System& s = system.value;
                return s.universe() ? s.universe()->cell_count() : 0;
            })
        // Per-particle state as owned numpy arrays (a host copy of the live prefix).
        .def("positions", &column_to_numpy<FluidPositionState>,
             "Live particle positions as an (N, 3) float32 array.")
        .def("velocities", &column_to_numpy<FluidVelocityState>,
             "Live particle velocities as an (N, 3) float32 array.")
        .def(
            "species",
            [](const PySystem& system) {
                const System& s = system.value;
                const Fluid* fluid  = s.fluid().get();
                const auto* state   = fluid ? fluid->state<FluidSpeciesState>() : nullptr;
                const std::size_t n = fluid ? fluid->particle_count() : 0;

                auto* buffer = new std::uint64_t[n];
                if (state != nullptr) {
                    const auto& device = state->data();
                    const atlas::HostBuffer<std::size_t> host(device.begin(), device.end());
                    for (std::size_t i = 0; i < n; ++i) {
                        buffer[i] = static_cast<std::uint64_t>(host[i]);
                    }
                }
                nb::capsule owner(buffer,
                                  [](void* p) noexcept { delete[] static_cast<std::uint64_t*>(p); });
                return nb::ndarray<nb::numpy, std::uint64_t, nb::shape<-1>>(buffer, { n }, owner);
            },
            "Live per-particle species ids as an (N,) uint64 array.");

    nb::class_<Observer>(m, "Observer");

    m.def(
        "observer",
        [](const std::size_t interval, const std::filesystem::path& output_directory) {
            return Observer::builder()
                .with_interval(interval)
                .with_output_directory(output_directory)
                .make_host_shared();
        },
        "interval"_a, "output_directory"_a,
        "A CSV observer sampling every `interval` steps into `output_directory`.");

    // A single assembly entry point. Fluid and Universe are unique_ptr owners and
    // are moved in; the policy objects are shared/value handles the other groups
    // hand out. Optional subsystems default to empty and are simply not wired.
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
           ObserverHostPtr observer) {
            System::Builder builder = System::builder();
            builder.with_fluid(std::move(fluid));
            builder.with_universe(std::move(universe));
            builder.with_dt(dt);

            if (solver) {
                builder.with_solver(std::move(solver));
            }
            // The emitter is a source/generator pair; wire it only when both are given.
            MeshOwners mesh_owners;
            if (source.has_value() && source->value && generator) {
                builder.with_emitter(std::move(source->value), std::move(generator));
                mesh_owners.insert(mesh_owners.end(),
                                   source->mesh_owners.begin(),
                                   source->mesh_owners.end());
            }
            for (const PyCollider& collider : colliders) {
                builder.with_collider(collider.value);
                mesh_owners.insert(mesh_owners.end(),
                                   collider.mesh_owners.begin(),
                                   collider.mesh_owners.end());
            }
            for (const PySink& sink : sinks) {
                builder.with_sink(sink.value);
                mesh_owners.insert(mesh_owners.end(),
                                   sink.mesh_owners.begin(),
                                   sink.mesh_owners.end());
            }
            if (codec) {
                builder.with_codec(std::move(codec));
            }
            if (observer) {
                builder.with_observer(std::move(observer));
            }

            return PySystem(builder.build(), std::move(mesh_owners));
        },
        "fluid"_a, "universe"_a, "dt"_a,
        "solver"_a    = SolverHostPtr {},
        "source"_a    = nb::none(),
        "generator"_a = GeneratorHostPtr {},
        "colliders"_a = std::vector<PyCollider> {},
        "sinks"_a     = std::vector<PySink> {},
        "codec"_a     = CodecHostPtr {},
        "observer"_a  = ObserverHostPtr {},
        "Assemble a runnable System from its subsystems (fluid and universe are consumed).");
}

}
