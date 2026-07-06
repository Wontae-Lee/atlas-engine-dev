#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cstddef>

namespace atlas {

class Box final {
public:
    class Builder;

public:
    Float3 lower_corner = Float3(-1.0f, -1.0f, -1.0f);
    Float3 upper_corner = Float3(1.0f, 1.0f, 1.0f);

    Box() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Box(const Float3& lower_corner_, const Float3& upper_corner_) noexcept
        : lower_corner(lower_corner_)
        , upper_corner(upper_corner_) { }

    Box(const Box& other) noexcept = default;
    Box(Box&& other) noexcept      = default;
    Box&
    operator=(const Box& other) noexcept = default;
    Box&
    operator=(Box&& other) noexcept = default;

    ~Box() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        Float3 cp = atlas::clamp(p, lower_corner, upper_corner);

        const bool inside = atlas::all(p >= lower_corner)
            && atlas::all(p <= upper_corner);

        if (inside) {
            bool hit_lower;
            const std::size_t axis = nearest_face(p, hit_lower);

            cp[axis] = hit_lower ? lower_corner[axis] : upper_corner[axis];
        }

        return cp;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3& p) const noexcept {
        const bool inside = atlas::all(p >= lower_corner)
            && atlas::all(p <= upper_corner);

        Float3 n(0.0f);

        if (inside) {
            bool hit_lower;
            const std::size_t axis = nearest_face(p, hit_lower);

            n[axis] = hit_lower ? -1.0f : 1.0f;
            return n;
        }

        const Float3 cp = atlas::clamp(p, lower_corner, upper_corner);
        const Float3 d  = p - cp;

        const std::size_t axis = atlas::abs(d).major_axis();

        n[axis] = (d[axis] >= 0.0f) ? 1.0f : -1.0f;
        return n;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const bool inside = atlas::all(p >= lower_corner)
            && atlas::all(p <= upper_corner);

        if (inside) {
            const Float3 l_to_p = p - lower_corner;
            const Float3 p_to_u = upper_corner - p;

            const float m1 = l_to_p.min();
            const float m2 = p_to_u.min();

            return -((m1 < m2) ? m1 : m2);
        }

        const Float3 cp = atlas::clamp(p, lower_corner, upper_corner);
        return (cp - p).length();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        return atlas::all(p >= lower_corner - tolerance)
            && atlas::all(p <= upper_corner + tolerance);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (tolerance < 0.0f) {
            return false;
        }

        const Float3& lo       = lower_corner;
        const Float3& hi       = upper_corner;
        const float tolerance2 = tolerance * tolerance;

        const bool inside = atlas::all(p >= lo)
            && atlas::all(p <= hi);

        if (inside) {
            const float dx = (p.x - lo.x < hi.x - p.x) ? p.x - lo.x : hi.x - p.x;
            const float dy = (p.y - lo.y < hi.y - p.y) ? p.y - lo.y : hi.y - p.y;
            const float dz = (p.z - lo.z < hi.z - p.z) ? p.z - lo.z : hi.z - p.z;
            const float d  = (dx < dy) ? ((dx < dz) ? dx : dz)
                                       : ((dy < dz) ? dy : dz);

            return d <= tolerance;
        }

        const float dx = p.x < lo.x ? lo.x - p.x
            : p.x > hi.x            ? p.x - hi.x
                                    : 0.0f;
        const float dy = p.y < lo.y ? lo.y - p.y
            : p.y > hi.y            ? p.y - hi.y
                                    : 0.0f;
        const float dz = p.z < lo.z ? lo.z - p.z
            : p.z > hi.z            ? p.z - hi.z
                                    : 0.0f;

        return dx * dx + dy * dy + dz * dz <= tolerance2;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return (lower_corner + upper_corner) * 0.5f;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        return AABB(lower_corner, upper_corner);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return atlas::isfinite(lower_corner)
            && atlas::isfinite(upper_corner)
            && atlas::all(upper_corner >= lower_corner);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& r) const noexcept {
        HitSurface result {};

        const HitAABB hit = bound().trace(r);

        if (!hit.is_intersecting || hit.exit < eps) {
            return result;
        }

        const bool use_enter = (hit.enter >= eps);
        const float t        = use_enter ? hit.enter : hit.exit;

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = r.point_at(t);
        result.normal          = closest_normal(result.point);

        return result;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept {
        return trace(ray);
    }

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    nearest_face(const Float3& p, bool& hit_lower) const noexcept {
        const Float3 l_to_p = p - lower_corner;
        const Float3 p_to_u = upper_corner - p;

        hit_lower = (l_to_p.min() < p_to_u.min());

        return hit_lower ? l_to_p.minor_axis() : p_to_u.minor_axis();
    }
};

class Box::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Box
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Box>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_lower_corner(const Float3& lower_corner_) noexcept;

    ATLAS_HOST Builder&
    with_upper_corner(const Float3& upper_corner_) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    Float3 _lower_corner = Float3(-1.0f, -1.0f, -1.0f);
    Float3 _upper_corner = Float3(1.0f, 1.0f, 1.0f);
};

using BoxHostPtr = atlas::host_shared_ptr<Box>;

using BoxDevicePtr = atlas::device_shared_ptr<Box>;

}
