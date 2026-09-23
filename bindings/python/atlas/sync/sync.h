#pragma once

#include <atlas/math/quaternion.h>
#include <atlas/math/vector/float3.h>
#include <atlas/spatial/ray.h>
#include <atlas/sync/sync.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_sync(nb::module_& m) {
    auto type = nb::class_<Sync>(m, "Sync")
        .def_prop_rw(
            "translation",
            [](const Sync& pose) { return pose.translation; },
            &Sync::set_translation)
        .def_prop_rw(
            "orientation",
            [](const Sync& pose) { return pose.orientation; },
            &Sync::set_orientation)
        .def_prop_ro("orientation_matrix", [](const Sync& pose) { return pose.orientation_matrix; })
        .def_prop_ro("inverse_orientation_matrix", [](const Sync& pose) {
            return pose.inverse_orientation_matrix;
        })
        .def("set_translation", &Sync::set_translation, "translation"_a)
        .def("set_orientation", &Sync::set_orientation, "orientation"_a)
        .def("set_pose", &Sync::set_pose, "translation"_a, "orientation"_a)
        .def("rebuild_matrices", &Sync::rebuild_matrices)
        .def(
            "sync_to_world",
            [](const Sync& pose, const Float3& point) {
                return pose.sync_to_world(point);
            },
            "point"_a)
        .def(
            "sync_to_world",
            [](const Sync& pose, const Ray& ray) {
                return pose.sync_to_world(ray);
            },
            "ray"_a)
        .def(
            "sync_to_local",
            [](const Sync& pose, const Float3& point) {
                return pose.sync_to_local(point);
            },
            "point"_a)
        .def(
            "sync_to_local",
            [](const Sync& pose, const Ray& ray) {
                return pose.sync_to_local(ray);
            },
            "ray"_a)
        .def(
            "sync_dir_to_world",
            [](const Sync& pose, const Float3& direction) {
                return pose.sync_dir_to_world(direction);
            },
            "direction"_a)
        .def(
            "sync_dir_to_local",
            [](const Sync& pose, const Float3& direction) {
                return pose.sync_dir_to_local(direction);
            },
            "direction"_a);

    type.def(nb::new_([](const Float3& translation, const Quaternion& orientation) {
            return Sync::builder().with_rigid_pose(translation, orientation).make_host_shared();
        }),
        "translation"_a = Float3(),
        "orientation"_a = Quaternion(),
        "A rigid-body pose (local-to-world transform) as a shared handle; the "
        "orientation defaults to the identity rotation.");
}

}
