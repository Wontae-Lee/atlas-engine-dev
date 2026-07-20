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

// The rigid-body transform layer. A Sync is a local-to-world pose (translation +
// orientation) and a Unit couples a Geometry with a Sync and its optional
// kinematics. As with geometry, we expose small factory functions that run the
// fluent builders and hand back ready-to-use values, keeping the Python surface
// flat and sidestepping the builders' reference lifetimes.
namespace atlas::python {

void
register_transform(nb::module_& m) {
    // Opaque value type: a Sync is consumed by `unit` and never inspected from
    // Python, so no methods are exposed.
    nb::class_<Sync>(m, "Sync");

    m.def(
        "sync",
        [](const Float3& translation, const Quaternion& orientation) {
            return Sync::builder().with_rigid_pose(translation, orientation).make_host_shared();
        },
        "translation"_a, "orientation"_a = Quaternion(),
        "A rigid-body pose (local-to-world transform) as a shared handle; the "
        "orientation defaults to the identity rotation.");

    // Opaque value type: a Unit is assembled here and handed to boundary code.
    nb::class_<PyUnit>(m, "Unit");

    m.def(
        "unit",
        [](const PyGeometry& geometry,
           SyncHostPtr sync,
           std::optional<Float3> velocity,
           std::optional<Float3> angular_velocity,
           std::optional<Float3> acceleration,
           std::optional<Float3> angular_acceleration) {
            auto b = Unit::builder().with_geometry(geometry.value);

            // A null handle means "place at the identity pose".
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
