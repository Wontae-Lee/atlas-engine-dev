#include "register.h"
#include "binding_types.h"

#include <atlas/geometry/geometry.h>
#include <atlas/math/quaternion.h>
#include <atlas/math/vector/float3.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <optional>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

void
register_transform(nb::module_& m) {
    nb::class_<Sync>(m, "Sync")
        .def(nb::init<>())
        .def(nb::init<const Sync&>(), "other"_a)
        .def_prop_rw("translation", [](const Sync& pose) { return pose.translation; },
                     &Sync::set_translation)
        .def_prop_rw("orientation", [](const Sync& pose) { return pose.orientation; },
                     &Sync::set_orientation)
        .def_prop_ro("orientation_matrix", [](const Sync& pose) { return pose.orientation_matrix; })
        .def_prop_ro("inverse_orientation_matrix", [](const Sync& pose) {
            return pose.inverse_orientation_matrix;
        })
        .def("set_translation", &Sync::set_translation, "translation"_a)
        .def("set_orientation", &Sync::set_orientation, "orientation"_a)
        .def("set_pose", &Sync::set_pose, "translation"_a, "orientation"_a)
        .def("rebuild_matrices", &Sync::rebuild_matrices)
        .def("sync_to_world", [](const Sync& pose, const Float3& point) {
            return pose.sync_to_world(point);
        }, "point"_a)
        .def("sync_to_world", [](const Sync& pose, const Ray& ray) {
            return pose.sync_to_world(ray);
        }, "ray"_a)
        .def("sync_to_local", [](const Sync& pose, const Float3& point) {
            return pose.sync_to_local(point);
        }, "point"_a)
        .def("sync_to_local", [](const Sync& pose, const Ray& ray) {
            return pose.sync_to_local(ray);
        }, "ray"_a)
        .def("sync_dir_to_world", [](const Sync& pose, const Float3& direction) {
            return pose.sync_dir_to_world(direction);
        }, "direction"_a)
        .def("sync_dir_to_local", [](const Sync& pose, const Float3& direction) {
            return pose.sync_dir_to_local(direction);
        }, "direction"_a);

    m.def(
        "sync",
        [](const Float3& translation, const Quaternion& orientation) {
            return Sync::builder().with_rigid_pose(translation, orientation).make_host_shared();
        },
        "translation"_a = Float3(), "orientation"_a = Quaternion(),
        "A rigid-body pose (local-to-world transform) as a shared handle; the "
        "orientation defaults to the identity rotation.");

    nb::class_<PyUnit>(m, "Unit")
        .def(nb::init<>())
        .def(nb::init<const PyUnit&>(), "other"_a)
        .def("update", [](PyUnit& unit, float dt) { unit.value.update(dt); }, "dt"_a)
        .def("move", [](PyUnit& unit, const Float3& delta) { unit.value.move(delta); }, "delta"_a)
        .def("rotate", [](PyUnit& unit, const Float3& axis, float radians) {
            unit.value.rotate(axis, radians);
        }, "axis"_a, "radians"_a)
        .def("set_geometry", [](PyUnit& unit, const PyGeometry& geometry) {
            auto owners = geometry.mesh_owners;
            unit.value.set_geometry(geometry.value);
            unit.mesh_owners = std::move(owners);
        }, "geometry"_a)
        .def("set_sync", [](PyUnit& unit, const Sync& pose) { unit.value.set_sync(pose); }, "sync"_a)
        .def_prop_ro("geometry", [](const PyUnit& unit) {
            return PyGeometry { unit.value.geometry(), unit.mesh_owners };
        })
        .def_prop_ro("sync", [](const PyUnit& unit) { return unit.value.sync(); })
        .def_prop_ro("velocity", [](const PyUnit& unit) { return unit.value.velocity(); })
        .def_prop_ro("acceleration", [](const PyUnit& unit) { return unit.value.acceleration(); })
        .def_prop_ro("angular_velocity", [](const PyUnit& unit) { return unit.value.angular_velocity(); })
        .def_prop_ro("angular_acceleration", [](const PyUnit& unit) { return unit.value.angular_acceleration(); })
        .def_prop_ro("dynamic", [](const PyUnit& unit) { return unit.value.dynamic(); })
        .def("trace", [](const PyUnit& unit, const Ray& ray) { return unit.value.trace(ray); }, "ray"_a)
        .def("surface_velocity", [](const PyUnit& unit, const Float3& point) {
            return unit.value.surface_velocity(point);
        }, "surface_point"_a)
        .def("world_bound", [](const PyUnit& unit) { return unit.value.world_bound(); });

    m.def("transform_aabb", [](const AABB& bound, const Sync& pose) {
        return atlas::transform_aabb(bound, [&pose](const Float3& point) {
            return pose.sync_to_world(point);
        });
    }, "bound"_a, "sync"_a);

    m.def("transform_aabb", [](const AABB& bound, const nb::callable& transform) {
        AABB transformed;
        if (!bound.is_valid()) return transformed;
        for (std::size_t corner = 0; corner < 8; ++corner)
            transformed.merge(nb::cast<Float3>(transform(bound.corner(corner))));
        return transformed;
    }, "bound"_a, "transform"_a);

    m.def(
        "unit",
        [](const PyGeometry& geometry,
           SyncHostPtr sync,
           std::optional<Float3> velocity,
           std::optional<Float3> angular_velocity,
           std::optional<Float3> acceleration,
           std::optional<Float3> angular_acceleration) {
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
        },
        "geometry"_a, "sync"_a = SyncHostPtr(), "velocity"_a = nb::none(),
        "angular_velocity"_a = nb::none(), "acceleration"_a = nb::none(),
        "angular_acceleration"_a = nb::none(),
        "A placeable rigid body: a Geometry at a Sync pose, optionally moving. "
        "sync defaults to the identity pose; kinematic values are left unset when omitted.");
}

}
