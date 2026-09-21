#pragma once

#include "../_detail/serialization.h"

namespace atlas::python {

inline void
register_protobuf_snapshot(nb::module_& m) {
    m.def(
        "save_fluid_binary",
        [](const Fluid& fluid, const std::filesystem::path& path) {
            save_fluid_binary(fluid, path.string());
        },
        "fluid"_a,
        "path"_a);

    m.def(
        "save_fluid_binary",
        [](const PySystem& system, const std::filesystem::path& path) {
            save_fluid_binary(*system.value.fluid(), path.string());
        },
        "system"_a,
        "path"_a);

    m.def(
        "save_universe_binary",
        [](const Universe& universe, const std::filesystem::path& path) {
            save_universe_binary(universe, path.string());
        },
        "universe"_a,
        "path"_a);

    m.def(
        "save_universe_binary",
        [](const PySystem& system, const std::filesystem::path& path) {
            save_universe_binary(*system.value.universe(), path.string());
        },
        "system"_a,
        "path"_a);

    m.def(
        "load_fluid_binary",
        [](const std::filesystem::path& path) {
            const FluidBinarySnapshot snapshot = load_fluid_binary(path.string());
            nb::dict result;
            result["buffer_size"]        = snapshot.buffer_size;
            result["particle_count"]     = snapshot.particle_count;
            result["statistical_weight"] = snapshot.statistical_weight;
            nb::list materials;
            for (const Material& material : snapshot.materials) {
                materials.append(nb::cast(material, nb::rv_policy::copy));
            }
            result["materials"]            = materials;
            result["positions"]            = snapshot_array(snapshot.positions);
            result["velocities"]           = snapshot_array(snapshot.velocities);
            result["species"]              = snapshot_array(snapshot.species);
            result["active"]               = snapshot_array(snapshot.active);
            result["temperature"]          = snapshot_array(snapshot.temperature);
            result["translational_energy"] = snapshot_array(snapshot.translational_energy);
            result["rotational_energy"]    = snapshot_array(snapshot.rotational_energy);
            result["vibrational_energy"]   = snapshot_array(snapshot.vibrational_energy);
            return result;
        },
        "path"_a,
        "Decode a fluid snapshot into metadata and owned arrays covering its full capacity.");

    m.def(
        "load_universe_binary",
        [](const std::filesystem::path& path) {
            const UniverseBinarySnapshot snapshot = load_universe_binary(path.string());
            nb::dict result;
            result["lower_corner"]       = nb::cast(snapshot.lower_corner, nb::rv_policy::copy);
            result["upper_corner"]       = nb::cast(snapshot.upper_corner, nb::rv_policy::copy);
            result["cell_size"]          = snapshot.cell_size;
            result["temperature"]        = snapshot_array(snapshot.temperature);
            result["bulk_velocity"]      = snapshot_array(snapshot.bulk_velocity);
            result["field_force"]        = snapshot_array(snapshot.field_force);
            result["gravity"]            = snapshot_array(snapshot.gravity);
            result["max_relative_speed"] = snapshot_array(snapshot.max_relative_speed);
            result["max_sigma_g"]        = snapshot_array(snapshot.max_sigma_g);
            result["thermal_energy"]     = snapshot_array(snapshot.thermal_energy);
            result["number_particle"]    = snapshot_array(snapshot.number_particle);
            result["collision_count"]    = snapshot_array(snapshot.collision_count);
            result["knudsen_number"]     = snapshot_array(snapshot.knudsen_number);
            result["allocated_solver"]   = snapshot_array(snapshot.allocated_solver);
            return result;
        },
        "path"_a,
        "Decode a universe snapshot into grid geometry and owned arrays; absent states are None.");

    m.def(
        "restore_fluid",
        [](const std::filesystem::path& path) {
            return restore_fluid(path.string());
        },
        "path"_a,
        "Restore a device-resident Fluid from a binary snapshot.");

    m.def(
        "restore_universe",
        [](const std::filesystem::path& path) {
            return restore_universe(path.string());
        },
        "path"_a,
        "Restore a device-resident Universe from a binary snapshot.");
}

}
