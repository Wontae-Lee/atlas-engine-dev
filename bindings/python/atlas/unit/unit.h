#pragma once

#include "../detail/ownership.h"

#include <atlas/math/vector/float3.h>
#include <atlas/spatial/ray.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_unit(nb::module_& m) {
    auto type = nb::class_<PyUnit>(m, "Unit")
        .def(
            "update",
            [](PyUnit& unit, float dt) { unit.value.update(dt); },
            "dt"_a)
        .def(
            "move",
            [](PyUnit& unit, const Float3& delta) { unit.value.move(delta); },
            "delta"_a)
        .def(
            "rotate",
            [](PyUnit& unit, const Float3& axis, float radians) {
                unit.value.rotate(axis, radians);
            },
            "axis"_a,
            "radians"_a)
        .def(
            "set_geometry",
            [](PyUnit& unit, const PyGeometry& geometry) {
                auto owners = geometry.mesh_owners;
                unit.value.set_geometry(geometry.value);
                unit.mesh_owners = std::move(owners);
            },
            "geometry"_a)
        .def(
            "set_sync",
            [](PyUnit& unit, const Sync& pose) { unit.value.set_sync(pose); },
            "sync"_a)
        .def_prop_ro("geometry", [](const PyUnit& unit) {
            return PyGeometry { unit.value.geometry(), unit.mesh_owners };
        })
        .def_prop_ro("sync", [](const PyUnit& unit) { return unit.value.sync(); })
        .def_prop_ro("velocity", [](const PyUnit& unit) { return unit.value.velocity(); })
        .def_prop_ro("acceleration", [](const PyUnit& unit) { return unit.value.acceleration(); })
        .def_prop_ro("angular_velocity", [](const PyUnit& unit) { return unit.value.angular_velocity(); })
        .def_prop_ro("angular_acceleration", [](const PyUnit& unit) { return unit.value.angular_acceleration(); })
        .def_prop_ro("dynamic", [](const PyUnit& unit) { return unit.value.dynamic(); })
        .def(
            "trace",
            [](const PyUnit& unit, const Ray& ray) { return unit.value.trace(ray); },
            "ray"_a)
        .def(
            "surface_velocity",
            [](const PyUnit& unit, const Float3& point) {
                return unit.value.surface_velocity(point);
            },
            "surface_point"_a)
        .def("world_bound", [](const PyUnit& unit) { return unit.value.world_bound(); });

    type.def(nb::new_([](const PyGeometry& geometry,
           SyncHostPtr sync,
           std::optional<Float3>
               velocity,
           std::optional<Float3>
               angular_velocity,
           std::optional<Float3>
               acceleration,
           std::optional<Float3>
               angular_acceleration) {
            auto b = Unit::builder().with_geometry(geometry.value);

            if (sync) {
                b.with_sync(sync);
            } else {
                b.with_sync(Sync::builder().make_host_shared());
            }

            if (velocity.has_value()) {
                b.with_velocity(*velocity);
            }

            if (acceleration.has_value()) {
                b.with_acceleration(*acceleration);
            }

            if (angular_velocity.has_value()) {
                b.with_angular_velocity(*angular_velocity);
            }

            if (angular_acceleration.has_value()) {
                b.with_angular_acceleration(*angular_acceleration);
            }

            return PyUnit { b.build(), geometry.mesh_owners };
        }),
        "geometry"_a,
        "sync"_a                 = SyncHostPtr(),
        "velocity"_a             = nb::none(),
        "angular_velocity"_a     = nb::none(),
        "acceleration"_a         = nb::none(),
        "angular_acceleration"_a = nb::none(),
        "A placeable rigid body: a Geometry at a Sync pose, optionally moving. "
        "sync defaults to the identity pose; kinematic values are left unset when omitted.");
}

}
