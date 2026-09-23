#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/material/material.h>
#include <atlas/material/material_dictionary.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_material_dictionary(nb::module_& m) {
    auto type = nb::class_<MaterialDictionary>(m, "MaterialDictionary")
        .def("size", &MaterialDictionary::size)
        .def("empty", &MaterialDictionary::empty)
        .def("__len__", &MaterialDictionary::size)
        .def("__getitem__", [](const MaterialDictionary& dictionary, std::ptrdiff_t index) {
            const auto size = static_cast<std::ptrdiff_t>(dictionary.size());
            if (index < 0) index += size;
            if (index < 0 || index >= size) throw nb::index_error();
            const auto& materials = dictionary.materials();
            const HostBuffer<Material> selected(materials.begin() + index, materials.begin() + index + 1);
            return selected[0];
        })
        .def("__setitem__", [](MaterialDictionary& dictionary, std::ptrdiff_t index, const Material& material) {
            const auto size = static_cast<std::ptrdiff_t>(dictionary.size());
            if (index < 0) index += size;
            if (index < 0 || index >= size) throw nb::index_error();
            dictionary.materials()[static_cast<std::size_t>(index)] = material;
        })
        .def(
            "materials",
            [](const MaterialDictionary& dictionary) {
                const auto& materials = dictionary.materials();
                const HostBuffer<Material> host(materials.begin(), materials.end());
                return std::vector<Material>(host.begin(), host.end());
            },
            "Returns a host copy of the ordered material table.")
        .def(
            "set_materials",
            [](MaterialDictionary& dictionary, const std::vector<Material>& materials) {
                const HostBuffer<Material> host(materials.begin(), materials.end());
                dictionary.materials() = DeviceBuffer<Material>(host.begin(), host.end());
            },
            "materials"_a,
            "Replaces the device material table; existing particle species ids must remain valid.");

    type.def(nb::new_([](const std::vector<Material>& materials) {
            auto builder = MaterialDictionary::builder();
            for (const auto& material : materials) {
                builder.with_material(material);
            }
            return builder.make_host_shared();
        }),
        "materials"_a,
        "Builds a MaterialDictionary from an ordered list of Materials; each "
        "material's position becomes its species id. Returned as a shared pointer.");
}

}
