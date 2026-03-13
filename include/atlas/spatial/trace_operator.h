#pragma once

#include <atlas/container/container.h>
#include <atlas/geometry/geometry_type.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>
#include <atlas/spatial/ray.h>

namespace atlas::spatial {

/**
 * @brief Ray tracer for an axis-aligned box defined by two corners.
 *
 * @details
 * The underlying intersection solves three 1D slab intervals and intersects
 * them:
 * \f[
 *   t_{\mathrm{enter}} = \max_i \min(t_i^0, t_i^1), \qquad
 *   t_{\mathrm{exit}}  = \min_i \max(t_i^0, t_i^1).
 * \f]
 */
template <typename T>
struct BoxTraceOperator final {
    /// Pointer to the minimum box corner.
    const Vector3<T>* lower_corner = nullptr;
    /// Pointer to the maximum box corner.
    const Vector3<T>* upper_corner = nullptr;

    /// Trace a ray against the referenced box.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& r) const;
};

/**
 * @brief Ray tracer for a finite cylinder aligned with the local z-axis.
 *
 * @details
 * The side surface satisfies \f$x^2 + y^2 = r^2\f$ with axial clipping
 * \f$z \in [-h/2, h/2]\f$. The implementation tests the quadratic side hit and
 * the two planar caps, selecting the smallest non-negative solution.
 */
template <typename T>
struct CylinderTraceOperator final {
    /// Pointer to the cylinder center.
    const Vector3<T>* center = nullptr;
    /// Pointer to the cylinder radius.
    const T* radius          = nullptr;
    /// Pointer to the full cylinder height.
    const T* height          = nullptr;

    /// Trace a ray against the referenced finite cylinder.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& ray) const;
};

/**
 * @brief Ray tracer for a plane in implicit form.
 *
 * @details
 * The plane is represented as
 * \f[
 *   n \cdot x + d = 0,
 * \f]
 * where `normal` stores \f$n\f$ and `offset` stores \f$d\f$.
 */
template <typename T>
struct PlaneTraceOperator final {
    /// Pointer to the plane normal vector.
    const Vector3<T>* normal = nullptr;
    /// Pointer to the plane offset in the implicit equation.
    const T* offset          = nullptr;

    /// Trace a ray against the referenced plane.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& ray) const;
};

/**
 * @brief Ray tracer for a sphere.
 *
 * @details
 * The sphere equation is \f$\|x - c\|^2 = r^2\f$. Substituting the ray equation
 * yields a quadratic in \f$t\f$.
 */
template <typename T>
struct SphereTraceOperator final {
    /// Pointer to the sphere center.
    const Vector3<T>* center = nullptr;
    /// Pointer to the sphere radius.
    const T* radius          = nullptr;

    /// Trace a ray against the referenced sphere.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& ray) const;
};

/**
 * @brief Ray tracer for a single triangle.
 *
 * @details
 * The implementation uses the Moller-Trumbore test, solving barycentric
 * coordinates and the ray parameter simultaneously from
 * \f[
 *   O + tD = (1-u-v)A + uB + vC.
 * \f]
 */
template <typename T>
struct TriangleTraceOperator final {
    /// Pointer to triangle vertex A.
    const Vector3<T>* a      = nullptr;
    /// Pointer to triangle vertex B.
    const Vector3<T>* b      = nullptr;
    /// Pointer to triangle vertex C.
    const Vector3<T>* c      = nullptr;
    /// Optional pointer to a precomputed triangle normal.
    const Vector3<T>* normal = nullptr;

    /// Trace a ray against the referenced triangle.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& r) const;
};

/**
 * @brief Ray tracer over a triangle BVH.
 *
 * @details
 * Traversal first tests a ray against node AABBs and only executes expensive
 * triangle intersections inside candidate leaf nodes. The nearest valid hit
 * found so far acts as an upper bound on later AABB entry distances.
 */
template <typename T>
struct BvhTraceOperator final {
    /// Pointer to the BVH node array.
    const BVHNode<T>* nodes           = nullptr;
    /// Pointer to the primitive permutation array.
    const int* indices                = nullptr;
    /// Pointer to triangle storage.
    const TriangleContainer4<T>* tris = nullptr;
    /// Root node index.
    int root                          = -1;

    /// Trace a ray against the referenced BVH.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& r) const;
};

/**
 * @brief Tagged union dispatching to one concrete trace operator.
 *
 * @details
 * This wrapper stores exactly one active operator selected by `type`. It avoids
 * virtual dispatch in device code while still providing a single call surface.
 */
template <typename T>
struct TraceOperator {
    /// Active geometry tag.
    atlas::geometry::GeometryType type = atlas::geometry::GeometryType::Sphere;

    union {
        /// Sphere tracer.
        SphereTraceOperator<T> sphere;
        /// Cylinder tracer.
        CylinderTraceOperator<T> cylinder;
        /// Plane tracer.
        PlaneTraceOperator<T> plane;
        /// Box tracer.
        BoxTraceOperator<T> box;
        /// Triangle tracer.
        TriangleTraceOperator<T> triangle;
        /// BVH tracer.
        BvhTraceOperator<T> triangle_mesh;
    };

    // ---- special members (needed because union members may be non-trivial on some toolchains) ----
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    TraceOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    TraceOperator(const TraceOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE TraceOperator&
    operator=(const TraceOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~TraceOperator() noexcept;

    // ---- tagged constructors (host convenience) ----
    ATLAS_HOST
    TraceOperator(const SphereTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const CylinderTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const PlaneTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const BoxTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const TriangleTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const BvhTraceOperator<T>& op);

    // ---- call ----
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    trace(const Ray<T>& ray) const;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& ray) const;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const TraceOperator& other) noexcept;
};

} // namespace atlas::spatial

namespace atlas {
template <typename T>
using TraceOperator = spatial::TraceOperator<T>;
} // namespace atlas

#include <atlas/spatial/trace_operator.hpp>
