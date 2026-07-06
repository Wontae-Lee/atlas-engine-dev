#pragma once

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace atlas {

struct HitAABB {

    bool is_intersecting = false;

    float enter = 0.0f;

    float exit = std::numeric_limits<float>::max();
};

class AABB final {
public:
    Float3 lower_corner;

    Float3 upper_corner;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AABB() noexcept {
        reset();
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AABB(const Float3& point1, const Float3& point2) noexcept
        : lower_corner(cmin(point1, point2))
        , upper_corner(cmax(point1, point2)) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AABB(const AABB&) noexcept = default;

    AABB&
    operator=(const AABB&) noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    area() const noexcept {
        const float w = width();
        const float h = height();
        const float d = depth();

        return 2.0f * (w * h + w * d + h * d);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    width() const noexcept {
        return upper_corner.x - lower_corner.x;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    height() const noexcept {
        return upper_corner.y - lower_corner.y;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    depth() const noexcept {
        return upper_corner.z - lower_corner.z;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length(const std::size_t axis) const noexcept {
        return upper_corner[axis] - lower_corner[axis];
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    overlaps(const AABB& other) const noexcept {
        const bool separated = any((upper_corner < other.lower_corner) | (lower_corner > other.upper_corner));

        return !separated;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    contains(const Float3& point) const noexcept {
        return all((point >= lower_corner) & (point <= upper_corner));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    intersects(const Ray& ray) const noexcept {
        return trace(ray).is_intersecting;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitAABB
    trace(const Ray& ray) const noexcept {
        HitAABB isect {};

        float t_enter = 0.0f;
        float t_exit  = inf;

        for (int i = 0; i < 3; ++i) {
            const float o  = ray.origin[i];
            const float d  = ray.direction[i];
            const float mn = lower_corner[i];
            const float mx = upper_corner[i];

            if (std::abs(d) <= eps) {
                if (o < mn || o > mx) return isect;
                continue;
            }

            float t0 = (mn - o) / d;
            float t1 = (mx - o) / d;

            if (t0 > t1) {
                const float tmp = t0;
                t0              = t1;
                t1              = tmp;
            }

            t_enter = std::max(t_enter, t0);
            t_exit  = std::min(t_exit, t1);

            if (t_enter > t_exit) return isect;
        }

        if (contains(ray.origin)) t_enter = 0.0f;

        isect.is_intersecting = true;
        isect.enter           = t_enter;
        isect.exit            = t_exit;

        return isect;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    center() const noexcept {
        return (lower_corner + upper_corner) * 0.5f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    extents() const noexcept {
        return upper_corner - lower_corner;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    diagonal_length() const noexcept {
        return (upper_corner - lower_corner).length();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    diagonal_length_squared() const noexcept {
        return (upper_corner - lower_corner).length_squared();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return atlas::isfinite(lower_corner)
            && atlas::isfinite(upper_corner)
            && atlas::all(lower_corner <= upper_corner);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    reset() noexcept {
        lower_corner = Float3(inf, inf, inf);
        upper_corner = Float3(-inf, -inf, -inf);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const Float3& point) noexcept {
        lower_corner = cmin(lower_corner, point);
        upper_corner = cmax(upper_corner, point);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const AABB& other) noexcept {
        lower_corner = cmin(lower_corner, other.lower_corner);
        upper_corner = cmax(upper_corner, other.upper_corner);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    expand(const float delta) noexcept {
        const Float3 d(delta, delta, delta);

        lower_corner -= d;
        upper_corner += d;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    corner(const std::size_t idx) const noexcept {
        const float x = (idx & 1) ? upper_corner.x : lower_corner.x;
        const float y = (idx & 2) ? upper_corner.y : lower_corner.y;
        const float z = (idx & 4) ? upper_corner.z : lower_corner.z;

        return Float3(x, y, z);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    clamp(const Float3& point) const noexcept {
        return atlas::clamp(point, lower_corner, upper_corner);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_empty() const noexcept {
        return any(upper_corner <= lower_corner);
    }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
make_aabb(const Float3& p) noexcept {
    AABB b;
    b.lower_corner = p;
    b.upper_corner = p;
    return b;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
merge_aabb(const AABB& a,
           const AABB& b) noexcept {
    AABB out;
    out.lower_corner = cmin(a.lower_corner, b.lower_corner);
    out.upper_corner = cmax(a.upper_corner, b.upper_corner);
    return out;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
aabb_distance_squared(const AABB& bounds,
                      const Float3& point) noexcept {
    const Float3 closest = bounds.clamp(point);
    return (closest - point).length_squared();
}

template <typename TransformPoint>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
transform_aabb(const AABB& bound,
               TransformPoint transform) noexcept {
    AABB transformed {};
    if (!bound.is_valid()) {
        return transformed;
    }

    for (int corner = 0; corner < 8; ++corner) {
        transformed.merge(transform(bound.corner(static_cast<std::size_t>(corner))));
    }

    return transformed;
}

}
