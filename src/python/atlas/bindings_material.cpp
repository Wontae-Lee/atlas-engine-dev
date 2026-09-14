#include "register.h"

#include <atlas/buffer/host_buffer.h>
#include <atlas/material/atom.h>
#include <atlas/material/ion.h>
#include <atlas/material/material.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/material/molecule.h>
#include <atlas/material/neutron.h>
#include <atlas/material/solid.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// The `Material` umbrella wraps one species leaf (molecule/atom/ion/neutron/solid)
// and is the per-species value stored in a MaterialDictionary's device table.
// Rather than expose each leaf type and its constructor, we hand out factory
// functions that return a ready-to-use Material; the dictionary itself is
// move-only (it owns a DeviceBuffer), so it is only reachable through a shared
// pointer produced by its builder.
namespace atlas::python {

void
register_material(nb::module_& m) {
    nb::enum_<MaterialType>(m, "MaterialType")
        .value("molecule", MaterialType::molecule)
        .value("atom", MaterialType::atom)
        .value("ion", MaterialType::ion)
        .value("neutron", MaterialType::neutron)
        .value("solid", MaterialType::solid);

    nb::class_<Material>(m, "Material")
        .def_ro("type", &Material::type)
        .def("mass", &Material::mass)
        .def("translational_energy", &Material::translational_energy)
        .def("rotational_energy", &Material::rotational_energy)
        .def("vibrational_energy", &Material::vibrational_energy)
        .def("reference_diameter", &Material::reference_diameter)
        .def("reference_temperature", &Material::reference_temperature)
        .def("viscosity_index", &Material::viscosity_index)
        .def("scattering_parameter", &Material::scattering_parameter);

    m.def(
        "molecule",
        [](const float mass,
           const float translational_energy,
           const float rotational_energy,
           const float vibrational_energy,
           const float reference_diameter,
           const float reference_temperature,
           const float viscosity_index,
           const float scattering_parameter) {
            return Material(Molecule(mass,
                                     translational_energy,
                                     rotational_energy,
                                     vibrational_energy,
                                     reference_diameter,
                                     reference_temperature,
                                     viscosity_index,
                                     scattering_parameter));
        },
        "mass"_a, "translational_energy"_a, "rotational_energy"_a, "vibrational_energy"_a,
        "reference_diameter"_a, "reference_temperature"_a, "viscosity_index"_a,
        "scattering_parameter"_a,
        "A polyatomic (molecule) species wrapped as a Material.");

    m.def(
        "atom",
        [](const float mass,
           const float translational_energy,
           const float rotational_energy,
           const float vibrational_energy,
           const float reference_diameter,
           const float reference_temperature,
           const float viscosity_index,
           const float scattering_parameter) {
            return Material(Atom(mass,
                                 translational_energy,
                                 rotational_energy,
                                 vibrational_energy,
                                 reference_diameter,
                                 reference_temperature,
                                 viscosity_index,
                                 scattering_parameter));
        },
        "mass"_a, "translational_energy"_a, "rotational_energy"_a, "vibrational_energy"_a,
        "reference_diameter"_a, "reference_temperature"_a, "viscosity_index"_a,
        "scattering_parameter"_a,
        "A monatomic (atom) species wrapped as a Material.");

    m.def(
        "ion",
        [](const float mass,
           const float translational_energy,
           const float rotational_energy,
           const float vibrational_energy,
           const float reference_diameter,
           const float reference_temperature,
           const float viscosity_index,
           const float scattering_parameter) {
            return Material(Ion(mass,
                                translational_energy,
                                rotational_energy,
                                vibrational_energy,
                                reference_diameter,
                                reference_temperature,
                                viscosity_index,
                                scattering_parameter));
        },
        "mass"_a, "translational_energy"_a, "rotational_energy"_a, "vibrational_energy"_a,
        "reference_diameter"_a, "reference_temperature"_a, "viscosity_index"_a,
        "scattering_parameter"_a,
        "A charged (ion) species wrapped as a Material.");

    m.def(
        "neutron",
        [](const float mass,
           const float translational_energy,
           const float rotational_energy,
           const float vibrational_energy,
           const float reference_diameter,
           const float reference_temperature,
           const float viscosity_index,
           const float scattering_parameter) {
            return Material(Neutron(mass,
                                    translational_energy,
                                    rotational_energy,
                                    vibrational_energy,
                                    reference_diameter,
                                    reference_temperature,
                                    viscosity_index,
                                    scattering_parameter));
        },
        "mass"_a, "translational_energy"_a, "rotational_energy"_a, "vibrational_energy"_a,
        "reference_diameter"_a, "reference_temperature"_a, "viscosity_index"_a,
        "scattering_parameter"_a,
        "A neutral nuclear (neutron) species wrapped as a Material.");

    m.def(
        "solid",
        [](const float mass) { return Material(Solid(mass)); },
        "mass"_a,
        "An immobile boundary/wall (solid) species wrapped as a Material; only mass is carried.");

    nb::class_<MaterialDictionary>(m, "MaterialDictionary")
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
        .def("materials", [](const MaterialDictionary& dictionary) {
            const auto& materials = dictionary.materials();
            const HostBuffer<Material> host(materials.begin(), materials.end());
            return std::vector<Material>(host.begin(), host.end());
        }, "Returns a host copy of the ordered material table.")
        .def("set_materials", [](MaterialDictionary& dictionary, const std::vector<Material>& materials) {
            const HostBuffer<Material> host(materials.begin(), materials.end());
            dictionary.materials() = DeviceBuffer<Material>(host.begin(), host.end());
        }, "materials"_a, "Replaces the device material table; existing particle species ids must remain valid.");

    m.def(
        "material_dictionary",
        [](const std::vector<Material>& materials) {
            auto builder = MaterialDictionary::builder();
            for (const auto& material : materials) {
                builder.with_material(material);
            }
            return builder.make_host_shared();
        },
        "materials"_a,
        "Builds a MaterialDictionary from an ordered list of Materials; each "
        "material's position becomes its species id. Returned as a shared pointer.");
}

}
