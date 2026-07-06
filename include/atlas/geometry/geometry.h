#pragma once

#include <../core/device_variant.h>
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

namespace atlas {

template <typename S>
concept Shape = requires(const S s, const Float3 p, const Ray r, float tolerance) {
    { s.closest_point(p) } -> std::same_as<Float3>;
    { s.closest_normal(p) } -> std::same_as<Float3>;
    { s.signed_distance(p) } -> std::same_as<float>;
    { s.is_inside(p, tolerance) } -> std::same_as<bool>;
    { s.is_on_surface(p, tolerance) } -> std::same_as<bool>;
    { s.centroid() } -> std::same_as<Float3>;
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

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, float tolerance = 0.0f) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, float tolerance = 0.0f) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept;
};

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

struct GeometryClosestPoint {
    Float3 p;
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const S& geometry) const noexcept { return geometry.closest_point(p); }
};
struct GeometryClosestNormal {
    Float3 p;
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const S& geometry) const noexcept { return geometry.closest_normal(p); }
};
struct GeometrySignedDistance {
    Float3 p;
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const S& geometry) const noexcept { return geometry.signed_distance(p); }
};
struct GeometryIsInside {
    Float3 p;
    float tolerance;
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(const S& geometry) const noexcept { return geometry.is_inside(p, tolerance); }
};
struct GeometryIsOnSurface {
    Float3 p;
    float tolerance;
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(const S& geometry) const noexcept { return geometry.is_on_surface(p, tolerance); }
};
struct GeometryCentroid {
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const S& geometry) const noexcept { return geometry.centroid(); }
};
struct GeometryBound {
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    operator()(const S& geometry) const noexcept { return geometry.bound(); }
};
struct GeometryIsValid {
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(const S& geometry) const noexcept { return geometry.is_valid(); }
};
struct GeometryTrace {
    Ray ray;
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const S& geometry) const noexcept { return geometry.trace(ray); }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Geometry::Geometry() noexcept {
    GeometryVariant::construct(*this, GeometryType::sphere);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Geometry>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Geometry::Geometry(const Payload& op) {
    GeometryVariant::construct_payload(*this, op);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
Geometry::closest_point(const Float3& p) const noexcept {
    return GeometryVariant::visit(*this, GeometryClosestPoint { p }, p);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
Geometry::closest_normal(const Float3& p) const noexcept {
    return GeometryVariant::visit(
        *this,
        GeometryClosestNormal { p },
        Float3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Geometry::signed_distance(const Float3& p) const noexcept {
    return GeometryVariant::visit(
        *this,
        GeometrySignedDistance { p },
        std::numeric_limits<float>::infinity());
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Geometry::is_inside(const Float3& p, const float tolerance) const noexcept {
    return GeometryVariant::visit(*this, GeometryIsInside { p, tolerance }, false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Geometry::is_on_surface(const Float3& p, const float tolerance) const noexcept {
    return GeometryVariant::visit(*this, GeometryIsOnSurface { p, tolerance }, false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
Geometry::centroid() const noexcept {
    return GeometryVariant::visit(
        *this,
        GeometryCentroid {},
        Float3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
Geometry::bound() const noexcept {
    return GeometryVariant::visit(*this, GeometryBound {}, AABB());
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Geometry::is_valid() const noexcept {
    return GeometryVariant::visit(*this, GeometryIsValid {}, false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
Geometry::trace(const Ray& ray) const noexcept {
    return GeometryVariant::visit(*this, GeometryTrace { ray }, HitSurface {});
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
Geometry::operator()(const Ray& ray) const noexcept {
    return trace(ray);
}

}
