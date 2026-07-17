#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cmath>
#include <limits>

namespace atlas {

/**
 * @brief Closed regular polygonal prism aligned with the world z-axis.
 *
 * The xy cross-section is a regular polygon with @ref side_count vertices on
 * the circle of @ref radius. Its first vertex points along +x. The prism spans
 * `center.z +/- height/2` and includes both end caps.
 */
class PolygonalPrism final {
public:
    /// Host-side fluent builder that validates parameters before construction.
    class Builder;

public:
    Float3 center = Float3(0.0f); ///< Center of the prism.
    int side_count = 3;           ///< Number of polygon sides; must be at least three.
    float radius = 1.0f;          ///< Circumradius of the cross-section; must be positive.
    float height = 1.0f;          ///< Total extent along z; must be positive.

    /** @brief Constructs a unit triangular prism centered at the origin. */
    PolygonalPrism() noexcept = default;

    /**
     * @brief Constructs a regular polygonal prism.
     * @param center_ Center of the prism.
     * @param side_count_ Number of cross-section sides.
     * @param radius_ Cross-section circumradius.
     * @param height_ Total z extent.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    PolygonalPrism(const Float3& center_, int side_count_, float radius_, float height_) noexcept
        : center(center_)
        , side_count(side_count_)
        , radius(radius_)
        , height(height_) { }

    PolygonalPrism(const PolygonalPrism&) noexcept = default; ///< Trivial copy.
    PolygonalPrism(PolygonalPrism&&) noexcept = default;      ///< Trivial move.
    PolygonalPrism& operator=(const PolygonalPrism&) noexcept = default; ///< Trivial copy assignment.
    PolygonalPrism& operator=(PolygonalPrism&&) noexcept = default;      ///< Trivial move assignment.
    ~PolygonalPrism() noexcept = default; ///< Trivial destructor.

    /** @brief Returns a fresh validated host-side builder. @return A builder. */
    ATLAS_NODISCARD ATLAS_HOST static Builder builder() noexcept;

    /**
     * @brief Returns the nearest point on the closed prism surface.
     * @param p Query point in world space.
     * @return Nearest point on a side or cap.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        float qx;
        float qy;
        float nx;
        float ny;
        float polygon_distance;
        polygon_query(p, qx, qy, nx, ny, polygon_distance);

        const float hz = height * 0.5f;
        const float dz = p.z - center.z;
        const bool inside_xy = polygon_distance <= 0.0f;
        const bool inside_z = std::abs(dz) <= hz;

        if (inside_xy && inside_z) {
            if (-polygon_distance <= hz - std::abs(dz)) {
                return Float3(qx, qy, p.z);
            }
            return Float3(p.x, p.y, center.z + (dz < 0.0f ? -hz : hz));
        }

        const float z = p.z < center.z - hz ? center.z - hz
            : p.z > center.z + hz ? center.z + hz : p.z;
        return Float3(inside_xy ? p.x : qx, inside_xy ? p.y : qy, z);
    }

    /**
     * @brief Returns the outward normal of the nearest surface feature.
     * @param p Query point in world space.
     * @return Unit side or cap normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3& p) const noexcept {
        const Float3 cp = closest_point(p);
        const float hz = height * 0.5f;
        if (std::abs(cp.z - (center.z - hz)) <= eps) {
            return Float3(0.0f, 0.0f, -1.0f);
        }
        if (std::abs(cp.z - (center.z + hz)) <= eps) {
            return Float3(0.0f, 0.0f, 1.0f);
        }

        float qx;
        float qy;
        float nx;
        float ny;
        float polygon_distance;
        polygon_query(p, qx, qy, nx, ny, polygon_distance);
        if (polygon_distance > 0.0f) {
            const float dx = p.x - qx;
            const float dy = p.y - qy;
            const float length = std::sqrt(dx * dx + dy * dy);
            if (length > 0.0f) {
                return Float3(dx / length, dy / length, 0.0f);
            }
        }
        return Float3(nx, ny, 0.0f);
    }

    /**
     * @brief Computes exact signed distance to the closed prism.
     * @param p Query point in world space.
     * @return Negative inside, zero on the surface, and positive outside.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        float qx;
        float qy;
        float nx;
        float ny;
        float polygon_distance;
        polygon_query(p, qx, qy, nx, ny, polygon_distance);
        const float axial = std::abs(p.z - center.z) - height * 0.5f;
        const float ox = polygon_distance > 0.0f ? polygon_distance : 0.0f;
        const float oz = axial > 0.0f ? axial : 0.0f;
        const float outside = std::sqrt(ox * ox + oz * oz);
        const float inside_max = polygon_distance > axial ? polygon_distance : axial;
        return outside + (inside_max < 0.0f ? inside_max : 0.0f);
    }

    /** @brief Tests containment after signed dilation. @param p Query point. @param tolerance Dilation. @return Whether signed distance is within tolerance. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, float tolerance = 0.0f) const noexcept {
        return signed_distance(p) <= tolerance;
    }

    /** @brief Tests a surface shell. @param p Query point. @param tolerance Non-negative shell half-width. @return Whether the point lies in the shell. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, float tolerance = 0.0f) const noexcept {
        return tolerance >= 0.0f && std::abs(signed_distance(p)) <= tolerance;
    }

    /** @brief Returns the geometric center. @return @ref center. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3 centroid() const noexcept { return center; }

    /** @brief Returns the axis-aligned bounds. @return Bounds using the circumradius in xy. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB bound() const noexcept {
        const float hz = height * 0.5f;
        return AABB(Float3(center.x - radius, center.y - radius, center.z - hz),
                    Float3(center.x + radius, center.y + radius, center.z + hz));
    }

    /** @brief Validates all shape parameters. @return True for finite center, radius, height, and at least three sides. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool is_valid() const noexcept {
        return side_count >= 3 && radius > 0.0f && height > 0.0f
            && atlas::isfinite(center) && atlas::isfinite(radius) && atlas::isfinite(height);
    }

    /**
     * @brief Intersects a ray by clipping it against every prism half-space.
     * @param ray World-space ray with normalized direction.
     * @return Nearest forward boundary hit, or an empty hit on a miss.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface hit {};
        float enter = -std::numeric_limits<float>::infinity();
        float exit = std::numeric_limits<float>::infinity();
        Float3 enter_normal(0.0f);
        Float3 exit_normal(0.0f);

        const float step = 2.0f * pi / static_cast<float>(side_count);
        for (int i = 0; i < side_count; ++i) {
            const float a = step * static_cast<float>(i);
            const float b = step * static_cast<float>(i + 1);
            const float ax = center.x + radius * std::cos(a);
            const float ay = center.y + radius * std::sin(a);
            const float bx = center.x + radius * std::cos(b);
            const float by = center.y + radius * std::sin(b);
            const float ex = bx - ax;
            const float ey = by - ay;
            const float inv = 1.0f / std::sqrt(ex * ex + ey * ey);
            const Float3 n(ey * inv, -ex * inv, 0.0f);
            if (!clip_plane(ray, Float3(ax, ay, center.z), n, enter, exit, enter_normal, exit_normal)) {
                return hit;
            }
        }

        const float hz = height * 0.5f;
        if (!clip_plane(ray, Float3(center.x, center.y, center.z + hz), Float3(0.0f, 0.0f, 1.0f), enter, exit, enter_normal, exit_normal)
            || !clip_plane(ray, Float3(center.x, center.y, center.z - hz), Float3(0.0f, 0.0f, -1.0f), enter, exit, enter_normal, exit_normal)) {
            return hit;
        }

        const bool use_enter = enter >= eps;
        const float distance = use_enter ? enter : exit;
        if (!(distance >= eps) || !atlas::isfinite(distance)) {
            return hit;
        }
        hit.is_intersecting = true;
        hit.distance = distance;
        hit.point = ray.point_at(distance);
        hit.normal = use_enter ? enter_normal : exit_normal;
        return hit;
    }

private:
    /**
     * @brief Finds the closest cross-section edge and signed 2D distance.
     * @param p Query point.
     * @param[out] qx Closest polygon x coordinate.
     * @param[out] qy Closest polygon y coordinate.
     * @param[out] nx Outward normal x coordinate of the nearest side plane.
     * @param[out] ny Outward normal y coordinate of the nearest side plane.
     * @param[out] distance Exact signed distance to the polygon boundary.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    polygon_query(const Float3& p, float& qx, float& qy, float& nx, float& ny, float& distance) const noexcept {
        const float step = 2.0f * pi / static_cast<float>(side_count);
        float closest2 = std::numeric_limits<float>::infinity();
        float maximum_plane = -std::numeric_limits<float>::infinity();
        nx = 1.0f;
        ny = 0.0f;

        for (int i = 0; i < side_count; ++i) {
            const float a = step * static_cast<float>(i);
            const float b = step * static_cast<float>(i + 1);
            const float ax = center.x + radius * std::cos(a);
            const float ay = center.y + radius * std::sin(a);
            const float bx = center.x + radius * std::cos(b);
            const float by = center.y + radius * std::sin(b);
            const float ex = bx - ax;
            const float ey = by - ay;
            const float edge2 = ex * ex + ey * ey;
            float t = ((p.x - ax) * ex + (p.y - ay) * ey) / edge2;
            t = t < 0.0f ? 0.0f : t > 1.0f ? 1.0f : t;
            const float cx = ax + t * ex;
            const float cy = ay + t * ey;
            const float dx = p.x - cx;
            const float dy = p.y - cy;
            const float d2 = dx * dx + dy * dy;
            const float inv = 1.0f / std::sqrt(edge2);
            const float edge_nx = ey * inv;
            const float edge_ny = -ex * inv;
            const float plane = (p.x - ax) * edge_nx + (p.y - ay) * edge_ny;

            if (d2 < closest2) {
                closest2 = d2;
                qx = cx;
                qy = cy;
            }
            if (plane > maximum_plane) {
                maximum_plane = plane;
                nx = edge_nx;
                ny = edge_ny;
            }
        }
        distance = maximum_plane <= 0.0f ? maximum_plane : std::sqrt(closest2);
    }

    /**
     * @brief Clips a ray interval against one inward half-space.
     * @param ray Query ray.
     * @param point Point on the clipping plane.
     * @param normal Outward plane normal.
     * @param[in,out] enter Current entry distance.
     * @param[in,out] exit Current exit distance.
     * @param[in,out] enter_normal Normal of the latest entry plane.
     * @param[in,out] exit_normal Normal of the earliest exit plane.
     * @return False when the clipped interval is empty.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    clip_plane(const Ray& ray, const Float3& point, const Float3& normal,
               float& enter, float& exit, Float3& enter_normal, Float3& exit_normal) noexcept {
        const float origin_distance = (ray.origin - point).dot(normal);
        const float direction = ray.direction.dot(normal);
        if (std::abs(direction) <= eps) {
            return origin_distance <= 0.0f;
        }
        const float t = -origin_distance / direction;
        if (direction < 0.0f) {
            if (t > enter) { enter = t; enter_normal = normal; }
        } else if (t < exit) {
            exit = t;
            exit_normal = normal;
        }
        return enter <= exit;
    }
};

/** @brief Host-side builder for validated regular polygonal prisms. */
class PolygonalPrism::Builder final {
public:
    Builder() = default; ///< Starts with a unit triangular prism.
    /** @brief Builds the prism. @return Validated prism. @throws std::runtime_error for invalid parameters. */
    ATLAS_NODISCARD ATLAS_HOST PolygonalPrism build() const;
    /** @brief Builds a shared host instance. @return Owning host pointer. @throws std::runtime_error for invalid parameters. */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<PolygonalPrism> make_host_shared() const;
    /** @brief Sets the center. @param center_ New center. @return `*this`. */
    ATLAS_HOST Builder& with_center(const Float3& center_) noexcept;
    /** @brief Sets the number of sides. @param side_count_ Side count of at least three. @return `*this`. */
    ATLAS_HOST Builder& with_side_count(int side_count_) noexcept;
    /** @brief Sets the circumradius. @param radius_ Positive radius. @return `*this`. */
    ATLAS_HOST Builder& with_radius(float radius_) noexcept;
    /** @brief Sets the total height. @param height_ Positive height. @return `*this`. */
    ATLAS_HOST Builder& with_height(float height_) noexcept;

private:
    /** @brief Rejects invalid accumulated parameters. @throws std::runtime_error when invalid. */
    ATLAS_HOST void validate() const;
    Float3 _center = Float3(0.0f); ///< Pending center.
    int _side_count = 3;           ///< Pending side count.
    float _radius = 1.0f;          ///< Pending circumradius.
    float _height = 1.0f;          ///< Pending height.
};

/// Owning host handle to a polygonal prism.
using PolygonalPrismHostPtr = atlas::host_shared_ptr<PolygonalPrism>;
/// Owning device handle to a polygonal prism.
using PolygonalPrismDevicePtr = atlas::device_shared_ptr<PolygonalPrism>;

}
