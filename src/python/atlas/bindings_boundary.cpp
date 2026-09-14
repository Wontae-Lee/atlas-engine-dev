#include "register.h"
#include "binding_types.h"

#include <atlas/collider/collider.h>
#include <atlas/collider/diffuse_sampling.h>
#include <atlas/collider/isothermal_collider.h>
#include <atlas/math/vector/float3.h>
#include <atlas/sink/sink.h>
#include <atlas/sink/surface_sink.h>
#include <atlas/sink/tracing_sink.h>
#include <atlas/sink/volume_sink.h>
#include <atlas/unit/unit.h>

#include <nanobind/nanobind.h>

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

// Boundary bindings: the `Collider` and `Sink` umbrellas. Both are copyable
// value types (the System builder takes them by `const&`), so — like Geometry —
// we hand out factory functions that return a ready-to-use umbrella rather than
// exposing each leaf and its fluent builder. A boundary owns a `Unit`, which the
// leaf builders take by value; the factories accept a Unit by value and move it in.
namespace atlas::python {

namespace {

struct BoundaryUnit final {
    template <typename Boundary>
    ATLAS_ALL_DEVICE Unit
    operator()(const Boundary& boundary) const noexcept {
        return boundary.unit();
    }
};

}

void
register_boundary(nb::module_& m) {
    // Hemisphere sampling law for the diffuse branch of an isothermal reflection.
    nb::enum_<DiffuseSampling>(m, "DiffuseSampling")
        .value("cosine_weighted", DiffuseSampling::cosine_weighted)
        .value("uniform", DiffuseSampling::uniform);

    nb::enum_<ColliderType>(m, "ColliderType")
        .value("isothermal", ColliderType::isothermal);
    nb::enum_<SinkType>(m, "SinkType")
        .value("surface", SinkType::surface)
        .value("volume", SinkType::volume)
        .value("tracing", SinkType::tracing);

    nb::class_<PyCollider>(m, "Collider")
        .def_prop_ro("type", [](const PyCollider& collider) { return collider.value.type; })
        .def_prop_ro("unit", [](const PyCollider& collider) {
            const Unit unit = ColliderVariant::visit(collider.value, BoundaryUnit {}, Unit {});
            return PyUnit { unit, collider.mesh_owners };
        })
        .def_prop_ro("momentum_accommodation_coefficient", [](const PyCollider& collider) {
            return collider.value.isothermal.momentum_accommodation_coefficient();
        })
        .def("bound", [](const PyCollider& collider) { return collider.value.bound(); })
        .def("advance", [](PyCollider& collider, const float dt) { collider.value.advance(dt); }, "dt"_a)
        .def("trace", [](const PyCollider& collider, const Float3& position,
                         const Float3& velocity, const float dt) {
            return collider.value.trace(position, velocity, dt);
        }, "position"_a, "velocity"_a, "dt"_a)
        .def("collide", [](const PyCollider& collider, const HitSurface& hit,
                           Float3 position, Float3 velocity, const float dt) {
            collider.value.collide(hit, position, velocity, dt);
            return nb::make_tuple(position, velocity);
        }, "hit"_a, "position"_a, "velocity"_a, "dt"_a,
        "Returns the post-collision position and velocity without modifying the input values.")
        .def("reflect", [](const PyCollider& collider, const Float3& incident, const Float3& normal) {
            return collider.value.isothermal.reflect(incident, normal);
        }, "incident"_a, "normal"_a);

    nb::class_<PySink>(m, "Sink")
        .def_prop_ro("type", [](const PySink& sink) { return sink.value.type; })
        .def_prop_ro("unit", [](const PySink& sink) {
            const Unit unit = SinkVariant::visit(sink.value, BoundaryUnit {}, Unit {});
            return PyUnit { unit, sink.mesh_owners };
        })
        .def("advance", [](PySink& sink, const float dt) { sink.value.advance(dt); }, "dt"_a)
        .def("despawn", [](const PySink& sink, const Float3& position,
                           const Float3& velocity, const float dt) {
            return sink.value.despawn(position, velocity, dt);
        }, "position"_a, "velocity"_a, "dt"_a);

    m.def(
        "isothermal_collider",
        [](PyUnit unit,
           const float momentum_accommodation_coefficient,
           const float restitution,
           const DiffuseSampling diffuse_sampling) {
            Collider collider(IsothermalCollider::builder()
                                .with_unit(std::move(unit.value))
                                .with_momentum_accommodation_coefficient(
                                    momentum_accommodation_coefficient)
                                .with_restitution(restitution)
                                .with_diffuse_sampling(diffuse_sampling)
                                .build());
            return PyCollider { std::move(collider), std::move(unit.mesh_owners) };
        },
        "unit"_a,
        "momentum_accommodation_coefficient"_a = 1.0f,
        "restitution"_a                        = 1.0f,
        "diffuse_sampling"_a                   = DiffuseSampling::uniform,
        "An isothermal moving-wall collider wrapped as a Collider.");

    m.def(
        "volume_sink",
        [](PyUnit unit, const float tolerance) {
            Sink sink(VolumeSink::builder()
                            .with_unit(std::move(unit.value))
                            .with_tolerance(tolerance)
                            .build());
            return PySink { std::move(sink), std::move(unit.mesh_owners) };
        },
        "unit"_a, "tolerance"_a = 0.0f,
        "A sink that despawns particles inside the unit's volume, wrapped as a Sink.");

    m.def(
        "surface_sink",
        [](PyUnit unit, const float tolerance) {
            Sink sink(SurfaceSink::builder()
                            .with_unit(std::move(unit.value))
                            .with_tolerance(tolerance)
                            .build());
            return PySink { std::move(sink), std::move(unit.mesh_owners) };
        },
        "unit"_a, "tolerance"_a = 0.0f,
        "A sink that despawns particles on the unit's surface, wrapped as a Sink.");

    m.def(
        "tracing_sink",
        [](PyUnit unit) {
            Sink sink(TracingSink::builder().with_unit(std::move(unit.value)).build());
            return PySink { std::move(sink), std::move(unit.mesh_owners) };
        },
        "unit"_a,
        "A swept sink that despawns particles crossing the unit this step, wrapped as a Sink.");
}

}
