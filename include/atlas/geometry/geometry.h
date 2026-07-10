#pragma once

#include <atlas/core/device_variant.h>
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

/**
 * @brief Compile-time contract every geometry leaf must satisfy.
 *
 * A type models `ConceptGeometry` when it exposes the full read-only surface
 * query interface that the `Geometry` umbrella forwards to. All queries are
 * `const` and callable on both host and device; the concept only checks the
 * signatures (name, arity and exact return type), not device-callability, which
 * the leaves guarantee by annotating each method `ATLAS_ALL_DEVICE`. Every
 * registered leaf is `static_assert`-ed against this concept immediately below,
 * so adding a leaf that drifts from the interface fails to compile here rather
 * than at the dispatch site.
 *
 * @tparam S Candidate geometry leaf type being tested for conformance.
 *
 * The required expressions, given a const `S`, a point `p`, a `Ray r` and a
 * `float tolerance`, are:
 * - `closest_point(p) -> Float3`   — nearest point on the surface to `p`.
 * - `closest_normal(p) -> Float3`  — outward unit normal at that nearest point.
 * - `signed_distance(p) -> float`  — distance to the surface, negative inside.
 * - `is_inside(p, tolerance) -> bool`    — containment test, `tolerance`-padded.
 * - `is_on_surface(p, tolerance) -> bool`— surface-shell test of half-width `tolerance`.
 * - `centroid() -> Float3`         — representative center of the shape.
 * - `bound() -> AABB`              — world-space axis-aligned bounding box.
 * - `is_valid() -> bool`           — whether the leaf's parameters are usable.
 * - `trace(r) -> HitSurface`       — ray/shape intersection.
 */
template <typename S>
concept ConceptGeometry = requires(const S s, const Float3 p, const Ray r, float tolerance) {
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

// Each leaf registered in the Geometry union must model the shared contract.
// Note that the triangle-mesh case is asserted through TriangleMeshView (the
// trivially-copyable view), not the buffer-owning TriangleMesh, because the
// union stores the view so the whole Geometry stays device-copyable.
static_assert(ConceptGeometry<Box>);
static_assert(ConceptGeometry<Circle>);
static_assert(ConceptGeometry<Cylinder>);
static_assert(ConceptGeometry<Plane>);
static_assert(ConceptGeometry<Sphere>);
static_assert(ConceptGeometry<Square>);
static_assert(ConceptGeometry<Triangle>);
static_assert(ConceptGeometry<TriangleMeshView>);

/**
 * @brief Tagged-union umbrella over every concrete geometry leaf.
 *
 * `Geometry` is a `DeviceVariant`: a `type` discriminant plus a `union` of the
 * self-contained leaf shapes. Because every leaf is trivially copyable, the
 * whole `Geometry` is trivially copyable and can be stored in a
 * `DeviceBuffer<Geometry>` and both dispatched and copied inside a device
 * lambda. All special members are defaulted so the compiler-generated copy is
 * the flat byte copy the device needs; the actual per-leaf routing is done by
 * the `GeometryVariant` alias declared below through placement-new and tagged
 * visitation.
 *
 * @note The triangle-mesh case holds a `TriangleMeshView` (a bundle of raw
 *       device pointers gathered on the host), not an owning mesh, so that the
 *       union never owns a `DeviceBuffer` and remains trivially copyable. The
 *       backing mesh storage lives elsewhere and must outlive this view.
 * @warning The union members are raw storage: read only the member selected by
 *          `type`. Constructing or reassigning must go through the
 *          `GeometryVariant` helpers (as the members below do) so the correct
 *          member is placement-new'd and `type` is kept consistent.
 */
struct Geometry {

    /// Active-leaf discriminant; selects which union member is live.
    GeometryType type = GeometryType::sphere;

    /**
     * @brief Storage for exactly one geometry leaf, selected by `type`.
     *
     * All members overlap; only the one named by `type` is alive at any time.
     */
    union {

        Box box; ///< Live when `type == GeometryType::box`.

        Circle circle; ///< Live when `type == GeometryType::circle`.

        Cylinder cylinder; ///< Live when `type == GeometryType::cylinder`.

        Plane plane; ///< Live when `type == GeometryType::plane`.

        Sphere sphere; ///< Live when `type == GeometryType::sphere` (default).

        Square square; ///< Live when `type == GeometryType::square`.

        Triangle triangle; ///< Live when `type == GeometryType::triangle`.

        TriangleMeshView triangle_mesh; ///< Live when `type == GeometryType::triangle_mesh`.
    };

    /**
     * @brief Constructs an empty default geometry: a unit `Sphere`.
     *
     * Routes through `GeometryVariant::construct(*this, GeometryType::sphere)`
     * so the `sphere` union member is placement-new'd and `type` set to
     * `sphere`. Callable on host and device.
     */
    ATLAS_ALL_DEVICE
    Geometry() noexcept;

    /**
     * @brief Trivial copy constructor: a flat byte copy of tag plus union.
     *
     * Defaulted on purpose so the copy is trivial and usable inside a device
     * lambda. Valid because every leaf is itself trivially copyable, so copying
     * the inactive-plus-active union bytes reproduces the source exactly.
     *
     * @param other Source geometry to copy.
     */
    ATLAS_ALL_DEVICE
    Geometry(const Geometry& other) noexcept = default;

    /**
     * @brief Trivial copy assignment: a flat byte copy of tag plus union.
     *
     * Defaulted for the same reason as the copy constructor; no old-leaf
     * destruction is needed because every leaf is trivially destructible.
     *
     * @param other Source geometry to copy from.
     * @return Reference to `*this`.
     */
    ATLAS_ALL_DEVICE Geometry&
    operator=(const Geometry& other) noexcept = default;

    /**
     * @brief Trivial destructor; leaves are trivially destructible so it is a no-op.
     */
    ATLAS_ALL_DEVICE
    ~Geometry() noexcept = default;

    /**
     * @brief Constructs a geometry directly from one concrete leaf value.
     *
     * Deduces the target union member from the payload's type via
     * `GeometryVariant::construct_payload`, placement-new'ing that member and
     * setting `type` accordingly. Passing an unsupported type is a compile-time
     * error inside the variant helper.
     *
     * @tparam Payload Concrete leaf type (e.g. `Box`, `Sphere`,
     *         `TriangleMeshView`); SFINAE-excluded when it is `Geometry` itself
     *         so this never shadows the copy constructor.
     * @param op The leaf value to store; copied into the union.
     */
    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Geometry>, int> = 0>
    ATLAS_ALL_DEVICE explicit Geometry(const Payload& op);

    /**
     * @brief Nearest point on the active leaf's surface to `p`.
     *
     * @param p Query point in world space.
     * @return The closest surface point; `Float3(0,0,0)` if no leaf is active.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept;

    /**
     * @brief Outward unit surface normal at the point nearest `p`.
     *
     * @param p Query point in world space.
     * @return Unit outward normal; `Float3(0,0,0)` fallback if no leaf is active.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3& p) const noexcept;

    /**
     * @brief Signed distance from `p` to the active leaf's surface.
     *
     * @param p Query point in world space.
     * @return Distance to the surface, negative inside the shape and positive
     *         outside; `+infinity` fallback if no leaf is active.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept;

    /**
     * @brief Tests whether `p` lies inside the active leaf.
     *
     * @param p Query point in world space.
     * @param tolerance Inflation of the shape before the test; positive values
     *        enlarge the containment region, negative values shrink it. Defaults
     *        to `0` for an exact test.
     * @return `true` if inside (as padded by `tolerance`); `false` fallback if
     *         no leaf is active.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, float tolerance = 0.0f) const noexcept;

    /**
     * @brief Tests whether `p` lies within a shell around the leaf's surface.
     *
     * @param p Query point in world space.
     * @param tolerance Half-thickness of the surface shell; the test accepts
     *        points within this band of the exact surface. Defaults to `0`.
     * @return `true` if on the surface within `tolerance`; `false` fallback if
     *         no leaf is active.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, float tolerance = 0.0f) const noexcept;

    /**
     * @brief Representative center of the active leaf.
     *
     * @return The leaf's centroid; `Float3(0,0,0)` fallback if no leaf is active.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept;

    /**
     * @brief World-space axis-aligned bounding box of the active leaf.
     *
     * @return The enclosing `AABB`; a default-constructed `AABB` fallback if no
     *         leaf is active.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept;

    /**
     * @brief Whether the active leaf's parameters describe a usable shape.
     *
     * @return `true` if the leaf reports valid parameters; `false` fallback if
     *         no leaf is active.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Intersects a ray with the active leaf.
     *
     * @param ray World-space ray (origin plus normalized direction).
     * @return The nearest forward hit; an empty `HitSurface`
     *         (`is_intersecting == false`) on a miss or if no leaf is active.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept;
};

/**
 * @brief Dispatch table binding each `GeometryType` tag to its union member.
 *
 * This `DeviceVariant` specialization is the engine that drives every
 * `Geometry` member function: it knows the owner type, the discriminant type,
 * the default tag (`sphere`), and one `DeviceVariantCase` per leaf mapping a tag
 * to the pointer-to-member it activates. `construct`, `construct_payload`,
 * `copy_construct` and `visit` all route through this alias so the tag and the
 * live union member never disagree.
 *
 * @note Any tag not listed here normalizes to `GeometryType::sphere`; all eight
 *       tags are registered, so normalization only matters for corrupt tags.
 */
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

/**
 * @brief Visitor forwarding `closest_point` to whichever leaf is active.
 *
 * A stateless-but-for-`p` functor callable object: `DeviceVariant::visit`
 * invokes `operator()` with the live leaf, and it forwards the stored query
 * point. Bundled as a struct rather than a lambda so it can be passed to the
 * variant's device-side visitation.
 */
struct GeometryClosestPoint {
    Float3 p; ///< World-space query point forwarded to the leaf.
    /**
     * @brief Forwards to `geometry.closest_point(p)`.
     * @tparam S Active leaf type deduced by the variant.
     * @param geometry The live union member.
     * @return Nearest surface point on that leaf.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const S& geometry) const noexcept { return geometry.closest_point(p); }
};
/**
 * @brief Visitor forwarding `closest_normal` to whichever leaf is active.
 */
struct GeometryClosestNormal {
    Float3 p; ///< World-space query point forwarded to the leaf.
    /**
     * @brief Forwards to `geometry.closest_normal(p)`.
     * @tparam S Active leaf type deduced by the variant.
     * @param geometry The live union member.
     * @return Outward unit normal at the nearest surface point.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const S& geometry) const noexcept { return geometry.closest_normal(p); }
};
/**
 * @brief Visitor forwarding `signed_distance` to whichever leaf is active.
 */
struct GeometrySignedDistance {
    Float3 p; ///< World-space query point forwarded to the leaf.
    /**
     * @brief Forwards to `geometry.signed_distance(p)`.
     * @tparam S Active leaf type deduced by the variant.
     * @param geometry The live union member.
     * @return Signed distance, negative inside the leaf.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const S& geometry) const noexcept { return geometry.signed_distance(p); }
};
/**
 * @brief Visitor forwarding `is_inside` to whichever leaf is active.
 */
struct GeometryIsInside {
    Float3 p;        ///< World-space query point forwarded to the leaf.
    float tolerance; ///< Containment padding forwarded to the leaf.
    /**
     * @brief Forwards to `geometry.is_inside(p, tolerance)`.
     * @tparam S Active leaf type deduced by the variant.
     * @param geometry The live union member.
     * @return Whether `p` is inside the leaf as padded by `tolerance`.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(const S& geometry) const noexcept { return geometry.is_inside(p, tolerance); }
};
/**
 * @brief Visitor forwarding `is_on_surface` to whichever leaf is active.
 */
struct GeometryIsOnSurface {
    Float3 p;        ///< World-space query point forwarded to the leaf.
    float tolerance; ///< Surface-shell half-thickness forwarded to the leaf.
    /**
     * @brief Forwards to `geometry.is_on_surface(p, tolerance)`.
     * @tparam S Active leaf type deduced by the variant.
     * @param geometry The live union member.
     * @return Whether `p` lies within `tolerance` of the leaf's surface.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(const S& geometry) const noexcept { return geometry.is_on_surface(p, tolerance); }
};
/**
 * @brief Visitor forwarding `centroid` to whichever leaf is active.
 */
struct GeometryCentroid {
    /**
     * @brief Forwards to `geometry.centroid()`.
     * @tparam S Active leaf type deduced by the variant.
     * @param geometry The live union member.
     * @return The leaf's representative center.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const S& geometry) const noexcept { return geometry.centroid(); }
};
/**
 * @brief Visitor forwarding `bound` to whichever leaf is active.
 */
struct GeometryBound {
    /**
     * @brief Forwards to `geometry.bound()`.
     * @tparam S Active leaf type deduced by the variant.
     * @param geometry The live union member.
     * @return The leaf's world-space bounding box.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    operator()(const S& geometry) const noexcept { return geometry.bound(); }
};
/**
 * @brief Visitor forwarding `is_valid` to whichever leaf is active.
 */
struct GeometryIsValid {
    /**
     * @brief Forwards to `geometry.is_valid()`.
     * @tparam S Active leaf type deduced by the variant.
     * @param geometry The live union member.
     * @return Whether the leaf's parameters are usable.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(const S& geometry) const noexcept { return geometry.is_valid(); }
};
/**
 * @brief Visitor forwarding `trace` to whichever leaf is active.
 */
struct GeometryTrace {
    Ray ray; ///< World-space ray forwarded to the leaf.
    /**
     * @brief Forwards to `geometry.trace(ray)`.
     * @tparam S Active leaf type deduced by the variant.
     * @param geometry The live union member.
     * @return The nearest forward hit on that leaf.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const S& geometry) const noexcept { return geometry.trace(ray); }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Geometry::Geometry() noexcept {
    // Activate the default sphere member and set the tag in one step.
    GeometryVariant::construct(*this, GeometryType::sphere);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Geometry>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Geometry::Geometry(const Payload& op) {
    // Deduce the union member from the payload's type and placement-new it there.
    GeometryVariant::construct_payload(*this, op);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
Geometry::closest_point(const Float3& p) const noexcept {
    // Fallback returns the query point itself (distance zero) when the tag matches no leaf.
    return GeometryVariant::visit(*this, GeometryClosestPoint { p }, p);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
Geometry::closest_normal(const Float3& p) const noexcept {
    // Fallback is a zero vector (no meaningful direction) for an unmatched tag.
    return GeometryVariant::visit(
        *this,
        GeometryClosestNormal { p },
        Float3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Geometry::signed_distance(const Float3& p) const noexcept {
    // Fallback of +infinity reads as "infinitely far / outside" for an unmatched tag.
    return GeometryVariant::visit(
        *this,
        GeometrySignedDistance { p },
        std::numeric_limits<float>::infinity());
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Geometry::is_inside(const Float3& p, const float tolerance) const noexcept {
    // Fallback of false: an unmatched tag contains nothing.
    return GeometryVariant::visit(*this, GeometryIsInside { p, tolerance }, false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Geometry::is_on_surface(const Float3& p, const float tolerance) const noexcept {
    // Fallback of false: an unmatched tag has no surface.
    return GeometryVariant::visit(*this, GeometryIsOnSurface { p, tolerance }, false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
Geometry::centroid() const noexcept {
    // Fallback is the origin when the tag matches no leaf.
    return GeometryVariant::visit(
        *this,
        GeometryCentroid {},
        Float3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
Geometry::bound() const noexcept {
    // Fallback is a default (empty) AABB for an unmatched tag.
    return GeometryVariant::visit(*this, GeometryBound {}, AABB());
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Geometry::is_valid() const noexcept {
    // Fallback of false: an unmatched tag is not a valid shape.
    return GeometryVariant::visit(*this, GeometryIsValid {}, false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
Geometry::trace(const Ray& ray) const noexcept {
    // Fallback is an empty HitSurface (is_intersecting == false) for an unmatched tag.
    return GeometryVariant::visit(*this, GeometryTrace { ray }, HitSurface {});
}

}