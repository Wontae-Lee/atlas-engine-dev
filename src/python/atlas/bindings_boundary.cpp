#include "register.h"

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

void
register_boundary(nb::module_& m) {
    // Hemisphere sampling law for the diffuse branch of an isothermal reflection.
    nb::enum_<DiffuseSampling>(m, "DiffuseSampling")
        .value("cosine_weighted", DiffuseSampling::cosine_weighted)
        .value("uniform", DiffuseSampling::uniform);

    // Opaque umbrellas: constructed only through the factories below.
    nb::class_<Collider>(m, "Collider");
    nb::class_<Sink>(m, "Sink");

    m.def(
        "isothermal_collider",
        [](Unit unit,
           const float momentum_accommodation_coefficient,
           const float restitution,
           const DiffuseSampling diffuse_sampling) {
            return Collider(IsothermalCollider::builder()
                                .with_unit(std::move(unit))
                                .with_momentum_accommodation_coefficient(
                                    momentum_accommodation_coefficient)
                                .with_restitution(restitution)
                                .with_diffuse_sampling(diffuse_sampling)
                                .build());
        },
        "unit"_a,
        "momentum_accommodation_coefficient"_a = 1.0f,
        "restitution"_a                        = 1.0f,
        "diffuse_sampling"_a                   = DiffuseSampling::uniform,
        "An isothermal moving-wall collider wrapped as a Collider.");

    m.def(
        "volume_sink",
        [](Unit unit, const float tolerance) {
            return Sink(VolumeSink::builder()
                            .with_unit(std::move(unit))
                            .with_tolerance(tolerance)
                            .build());
        },
        "unit"_a, "tolerance"_a = 0.0f,
        "A sink that despawns particles inside the unit's volume, wrapped as a Sink.");

    m.def(
        "surface_sink",
        [](Unit unit, const float tolerance) {
            return Sink(SurfaceSink::builder()
                            .with_unit(std::move(unit))
                            .with_tolerance(tolerance)
                            .build());
        },
        "unit"_a, "tolerance"_a = 0.0f,
        "A sink that despawns particles on the unit's surface, wrapped as a Sink.");

    m.def(
        "tracing_sink",
        [](Unit unit) {
            return Sink(TracingSink::builder().with_unit(std::move(unit)).build());
        },
        "unit"_a,
        "A swept sink that despawns particles crossing the unit this step, wrapped as a Sink.");
}

}
