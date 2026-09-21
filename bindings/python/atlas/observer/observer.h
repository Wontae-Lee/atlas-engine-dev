#pragma once

#include "../_detail/handles.h"
#include "../_detail/state.h"

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

inline void
register_observer(nb::module_& m) {
    auto type = nb::class_<Observer>(m, "Observer")
        .def_prop_ro("interval", &Observer::interval)
        .def_prop_ro("output_directory", &Observer::output_directory)
        .def_prop_ro("species_count", &Observer::species_count)
        .def("observe", &Observer::observe, "fluid"_a, "universe"_a, "step"_a)
        .def(
            "observe",
            [](const Observer& observer, const PySystem& system) {
                observer.observe(*system.value.fluid(), *system.value.universe(), system.value.step());
            },
            "system"_a)
        .def("resize_counters", &Observer::resize_counters, "source_count"_a, "sink_count"_a, "species_count"_a)
        .def("reset_counters", &Observer::reset_counters)
        .def("spawned", [](const Observer& observer) {
            const auto columns = observer.species_count();
            const auto rows    = columns ? observer.spawned().size() / columns : 0;
            return numpy_copy(observer.spawned(), observer.spawned().size()).attr("reshape")(rows, columns);
        })
        .def("despawned", [](const Observer& observer) {
            const auto columns = observer.species_count();
            const auto rows    = columns ? observer.despawned().size() / columns : 0;
            return numpy_copy(observer.despawned(), observer.despawned().size()).attr("reshape")(rows, columns);
        });

    type.def(nb::new_([](const std::size_t interval, const std::filesystem::path& output_directory) {
            return Observer::builder().with_interval(interval).with_output_directory(output_directory).make_host_shared();
        }),
        "interval"_a,
        "output_directory"_a,
        "A CSV observer sampling every `interval` steps into `output_directory`.");
}

}
