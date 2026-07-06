#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry_type.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/square.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>

#include <concepts>
#include <limits>
#include <type_traits>

/**
 * @file geometry.h
 * @brief Runtime-selectable, device-callable dispatcher over all eight
 *        shape operators (`Box`, ...,
 *        `TriangleMeshGeometryOperator`) — the value every device kernel
 *        actually queries geometry through; see `geometry.h`'s
 *        top-of-file documentation for why this exists separately from
 *        `Geometry`.
 *
 * @details
 * Same tagged-union `DeviceVariant` dispatch pattern as `DsmcKernel`/
 * `SphKernel`/`SurfaceInteractionKernel` (see those files): a
 * `GeometryType` tag selects which shape operator is active, and
 * `detail::GeometryVariant::visit` dispatches every query
 * (`closest_point`, `is_inside`, `trace`, ...) to it. Every shape
 * operator except `TriangleMeshGeometryOperator` value-embeds its
 * parameters (copied by value into the operator, no pointer into the
 * owning `Geometry` — see `docs/updates/updates.md` §2.5/§2.9 for the
 * historical dangling-pointer hazard this closed); `TriangleMeshGeometryOperator`
 * is the one exception, holding true pointer *views* into
 * vertex/index/BVH buffers owned by the mesh (a triangle mesh's data is
 * too large to value-embed per operator instance).
 */

namespace atlas {

/**
 * @brief Compile-time contract every shape stored in the `Geometry`
 *        union must satisfy — the device-callable query interface the
 *        variant dispatches to. Replaces the old abstract `Geometry`
 *        base's virtual interface with a concept (no vtables, so it
 *        works in device code), and the static_asserts below make a
 *        missing/mismatched method a clear per-shape error instead of an
 *        obscure failure deep inside `GeometryVariant::visit`.
 */
template <typename S>
concept Shape = requires(const S s, const Vector3 p, const Ray r, float tolerance) {
    { s.closest_point(p) } -> std::same_as<Vector3>;
    { s.closest_normal(p) } -> std::same_as<Vector3>;
    { s.signed_distance(p) } -> std::same_as<float>;
    { s.is_inside(p, tolerance) } -> std::same_as<bool>;
    { s.is_on_surface(p, tolerance) } -> std::same_as<bool>;
    { s.centroid() } -> std::same_as<Vector3>;
    { s.bound() } -> std::same_as<AABB>;
    { s.is_valid() } -> std::same_as<bool>;
    { s.trace(r) } -> std::same_as<HitSurface>;
};

static_assert(Shape<Box>);
static_assert(Shape<Circle>);
static_assert(Shape<Cylinder>);
static_assert(Shape<Plane>);
static_assert(Shape<Sphere>);
static_assert(Shape<Square>);
static_assert(Shape<Triangle>);
static_assert(Shape<TriangleMeshGeometryOperator>);

/**
 * @brief Tagged-union value type that holds one shape and dispatches
 *        device-callable queries (`closest_point`/`is_inside`/`trace`/
 *        ...) to it via a `GeometryType` tag — no virtual dispatch, so
 *        it is device-storable (a `Unit` holds one by value). Every union
 *        member is a `Shape` (see the concept above); the fixed-size
 *        shapes value-embed their parameters, while
 *        `TriangleMeshGeometryOperator` is a pointer view into the mesh's
 *        externally-owned buffers.
 */
struct Geometry {

    GeometryType type = GeometryType::sphere;

    union {

        Box box;

        Circle circle;

        Cylinder cylinder;

        Plane plane;

        Sphere sphere;

        Square square;

        Triangle triangle;

        TriangleMeshGeometryOperator triangle_mesh;
    };

    ATLAS_ALL_DEVICE
    Geometry() noexcept;

    ATLAS_ALL_DEVICE
    Geometry(const Geometry& other) noexcept = default;

    ATLAS_ALL_DEVICE Geometry&
    operator=(const Geometry& other) noexcept = default;

    ATLAS_ALL_DEVICE
    ~Geometry() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Geometry>, int> = 0>
    ATLAS_ALL_DEVICE explicit Geometry(const Payload& op);

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    closest_point(const Vector3& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    closest_normal(const Vector3& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Vector3& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const Vector3& p, float tolerance = 0.0f) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const Vector3& p, float tolerance = 0.0f) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /** @brief Ray-surface intersection query, dispatched to the active
     *  shape; used by `ColliderCollisionKernel`, `TracingDespawn`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept;

    /** @brief `trace(ray)`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept;
};

namespace detail {

    using GeometryVariant = DeviceVariant<
        Geometry,
        GeometryType,
        GeometryType::sphere,
        DeviceVariantCase<GeometryType::box, &Geometry::box>,
        DeviceVariantCase<GeometryType::circle, &Geometry::circle>,
        DeviceVariantCase<GeometryType::cylinder, &Geometry::cylinder>,
        DeviceVariantCase<GeometryType::plane, &Geometry::plane>,
        DeviceVariantCase<GeometryType::sphere, &Geometry::sphere>,
        DeviceVariantCase<GeometryType::square, &Geometry::square>,
        DeviceVariantCase<GeometryType::triangle, &Geometry::triangle>,
        DeviceVariantCase<GeometryType::triangle_mesh, &Geometry::triangle_mesh>>;

    // The variant visitors are functor structs rather than generic device
    // lambdas: nvcc forbids extended `__host__ __device__` lambdas that are
    // generic or capture by reference, whereas a struct with a templated
    // `operator()` and by-value members is unrestricted on device.
    struct GeometryClosestPoint {
        Vector3 p;
        template <typename S> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
        operator()(const S& geometry) const noexcept { return geometry.closest_point(p); }
    };
    struct GeometryClosestNormal {
        Vector3 p;
        template <typename S> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
        operator()(const S& geometry) const noexcept { return geometry.closest_normal(p); }
    };
    struct GeometrySignedDistance {
        Vector3 p;
        template <typename S> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
        operator()(const S& geometry) const noexcept { return geometry.signed_distance(p); }
    };
    struct GeometryIsInside {
        Vector3 p;
        float tolerance;
        template <typename S> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator()(const S& geometry) const noexcept { return geometry.is_inside(p, tolerance); }
    };
    struct GeometryIsOnSurface {
        Vector3 p;
        float tolerance;
        template <typename S> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator()(const S& geometry) const noexcept { return geometry.is_on_surface(p, tolerance); }
    };
    struct GeometryCentroid {
        template <typename S> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
        operator()(const S& geometry) const noexcept { return geometry.centroid(); }
    };
    struct GeometryBound {
        template <typename S> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
        operator()(const S& geometry) const noexcept { return geometry.bound(); }
    };
    struct GeometryIsValid {
        template <typename S> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator()(const S& geometry) const noexcept { return geometry.is_valid(); }
    };
    struct GeometryTrace {
        Ray ray;
        template <typename S> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
        operator()(const S& geometry) const noexcept { return geometry.trace(ray); }
    };

}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Geometry::Geometry() noexcept {
    detail::GeometryVariant::construct(*this, GeometryType::sphere);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Geometry>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Geometry::Geometry(const Payload& op) {
    detail::GeometryVariant::construct_payload(*this, op);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
Geometry::closest_point(const Vector3& p) const noexcept {
    return detail::GeometryVariant::visit(*this, detail::GeometryClosestPoint { p }, p);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
Geometry::closest_normal(const Vector3& p) const noexcept {
    return detail::GeometryVariant::visit(
        *this, detail::GeometryClosestNormal { p }, Vector3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Geometry::signed_distance(const Vector3& p) const noexcept {
    return detail::GeometryVariant::visit(
        *this, detail::GeometrySignedDistance { p }, std::numeric_limits<float>::infinity());
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Geometry::is_inside(const Vector3& p, const float tolerance) const noexcept {
    return detail::GeometryVariant::visit(*this, detail::GeometryIsInside { p, tolerance }, false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Geometry::is_on_surface(const Vector3& p, const float tolerance) const noexcept {
    return detail::GeometryVariant::visit(*this, detail::GeometryIsOnSurface { p, tolerance }, false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
Geometry::centroid() const noexcept {
    return detail::GeometryVariant::visit(
        *this, detail::GeometryCentroid {}, Vector3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
Geometry::bound() const noexcept {
    return detail::GeometryVariant::visit(*this, detail::GeometryBound {}, AABB());
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Geometry::is_valid() const noexcept {
    return detail::GeometryVariant::visit(*this, detail::GeometryIsValid {}, false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
Geometry::trace(const Ray& ray) const noexcept {
    return detail::GeometryVariant::visit(*this, detail::GeometryTrace { ray }, HitSurface {});
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
Geometry::operator()(const Ray& ray) const noexcept {
    return trace(ray);
}

}
