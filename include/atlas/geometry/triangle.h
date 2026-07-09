#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cmath>
#include <cstddef>
#include <optional>

namespace atlas {

class Triangle final {
public:
    class Builder;

public:
    Float3 a = Float3(0.0f, 0.0f, 0.0f);

    Float3 b = Float3(0.0f, 0.0f, 0.0f);

    Float3 c = Float3(0.0f, 0.0f, 0.0f);

    Float3 n = Float3(0.0f, 0.0f, 1.0f);

    Float3 normal = Float3(0.0f, 0.0f, 1.0f);

    Triangle() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Triangle(const Float3& a_, const Float3& b_, const Float3& c_) noexcept
        : a(a_)
        , b(b_)
        , c(c_) {
        normal = atlas::normalized_or(
            atlas::cross(b - a, c - a),
            Float3(0.0f, 0.0f, 0.0f));
    }

    Triangle(const Triangle& other) noexcept = default;
    Triangle(Triangle&& other) noexcept      = default;
    Triangle&
    operator=(const Triangle& other) noexcept = default;
    Triangle&
    operator=(Triangle&& other) noexcept = default;

    ~Triangle() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        const Float3 v0 = a;
        const Float3 v1 = b;
        const Float3 v2 = c;

        const Float3 ab = v1 - v0;
        const Float3 ac = v2 - v0;
        const Float3 ap = p - v0;

        const float d1 = ab.dot(ap);
        const float d2 = ac.dot(ap);

        if (d1 <= 0.0f && d2 <= 0.0f) {
            return v0;
        }

        const Float3 bp = p - v1;
        const float d3  = ab.dot(bp);
        const float d4  = ac.dot(bp);

        if (d3 >= 0.0f && d4 <= d3) {
            return v1;
        }

        const float vc = d1 * d4 - d3 * d2;

        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float vv = d1 / (d1 - d3);
            return v0 + ab * vv;
        }

        const Float3 cpv = p - v2;
        const float d5   = ab.dot(cpv);
        const float d6   = ac.dot(cpv);

        if (d6 >= 0.0f && d5 <= d6) {
            return v2;
        }

        const float vb = d5 * d2 - d1 * d6;

        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float ww = d2 / (d2 - d6);
            return v0 + ac * ww;
        }

        const float va = d3 * d6 - d5 * d4;

        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            const float ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return v1 + (v2 - v1) * ww;
        }

        const float denom = 1.0f / (va + vb + vc);
        const float vv    = vb * denom;
        const float ww    = vc * denom;

        return v0 + ab * vv + ac * ww;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3&) const noexcept {
        return normal;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const Float3 nn      = closest_normal(p);
        const float sd_plane = (p - a).dot(nn);

        const Float3 cp = closest_point(p);
        const float d   = (p - cp).length();

        return (sd_plane >= 0.0f) ? d : -d;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        const Float3 nn = normal;

        const float nn_len2 = nn.length_squared();

        if (nn_len2 <= 0.0f) {
            return false;
        }

        const float side = (p - a).dot(nn);

        if (side <= 0.0f) {
            if (tolerance >= 0.0f) {
                return true;
            }
        } else if (tolerance < 0.0f) {
            return false;
        }

        const Float3 cp = closest_point(p);
        const float d2  = (p - cp).length_squared();

        return side <= 0.0f ? d2 >= tolerance * tolerance
                            : d2 <= tolerance * tolerance;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (tolerance < 0.0f) {
            return false;
        }

        const Float3 cp = closest_point(p);
        return (p - cp).length_squared() <= tolerance * tolerance;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return (a + b + c) * (1.0f / 3.0f);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const Float3 mn = atlas::cmin(a, atlas::cmin(b, c));
        const Float3 mx = atlas::cmax(a, atlas::cmax(b, c));

        return AABB(mn, mx);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        const Float3 nn = atlas::cross(b - a, c - a);
        return nn.length_squared() > 0.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& r) const noexcept {
        HitSurface result {};

        const Float3 v0 = a;
        const Float3 v1 = b;
        const Float3 v2 = c;

        const Float3 e1 = v1 - v0;
        const Float3 e2 = v2 - v0;

        const Float3 pvec = atlas::cross(r.direction, e2);
        const float det   = e1.dot(pvec);

        if (std::abs(det) <= eps) {
            return result;
        }

        const float inv_det = 1.0f / det;

        const Float3 tvec = r.origin - v0;
        const float u     = tvec.dot(pvec) * inv_det;

        if (u < 0.0f || u > 1.0f) {
            return result;
        }

        const Float3 qvec = atlas::cross(tvec, e1);
        const float v     = r.direction.dot(qvec) * inv_det;

        if (v < 0.0f || (u + v) > 1.0f) {
            return result;
        }

        const float t = e2.dot(qvec) * inv_det;

        if (t < eps) {
            return result;
        }

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = r.point_at(t);

        const Float3 normal_vec = normal;

        result.normal = atlas::normalized_or(
            normal_vec,
            Float3(1.0f, 0.0f, 0.0f));

        return result;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept {
        return trace(ray);
    }
};

class Triangle::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Triangle
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Triangle>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_a(const Float3& a_) noexcept;

    ATLAS_HOST Builder&
    with_b(const Float3& b_) noexcept;

    ATLAS_HOST Builder&
    with_c(const Float3& c_) noexcept;

    ATLAS_HOST Builder&
    with_vertices(const Float3& a_, const Float3& b_, const Float3& c_) noexcept;

    ATLAS_HOST Builder&
    with_normal(const Float3& normal_) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    Float3 _a = Float3(0.0f, 0.0f, 0.0f);

    Float3 _b = Float3(0.0f, 0.0f, 0.0f);

    Float3 _c = Float3(0.0f, 0.0f, 0.0f);

    std::optional<Float3> _normal;
};

using TriangleHostPtr = atlas::host_shared_ptr<Triangle>;

using TriangleDevicePtr = atlas::device_shared_ptr<Triangle>;

}